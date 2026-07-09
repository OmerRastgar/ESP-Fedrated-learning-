import traceback
import os
import io
import time
import zipfile
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
EI_STUDIO_URL = "https://studio.edgeimpulse.com/v1"
EI_INGESTION_URL = "https://ingestion.edgeimpulse.com/api/training/data"

PROJECT_ID = int(os.getenv("EDGE_IMPULSE_PROJECT_ID", "0"))
API_KEY = os.getenv("EDGE_IMPULSE_API_KEY", "")

STATIC_MODEL_DIR = "./static_bin"
os.makedirs(STATIC_MODEL_DIR, exist_ok=True)

EXPECTED_CLASSES = {"STILL", "WALKING", "SHAKING", "TAPPING"}
uploaded_classes_registry = set()

# Pipeline state tracking for ESP32 polling
pipeline_state = {
    "building": False,
    "complete": False,
    "error": None,
    "message": "Waiting for data"
}

# ── HELPER: JOB POLLING LOGIC ──────────────────────────────────────────────────
def wait_for_job(job_id, job_name):
    """Polls the Edge Impulse API with network resilience and error logging."""
    headers = {"x-api-key": API_KEY}
    status_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/{job_id}/status"
    
    while True:
        try:
            res = requests.get(status_url, headers=headers, timeout=10)
            if res.status_code != 200:
                raise Exception(f"Failed to poll {job_name} job status. HTTP {res.status_code}")
                
            job_status = res.json().get("job", {})
            
            if job_status.get("finishedSuccessful"):
                print(f"  ✅ {job_name} complete!")
                return True
                
            elif job_status.get("finished"):
                # Job failed on the cloud side, let's fetch the logs
                log_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/{job_id}/log"
                try:
                    log_res = requests.get(log_url, headers=headers, timeout=10)
                    error_details = log_res.text if log_res.status_code == 200 else "Could not retrieve logs."
                    print(f"\n❌ {job_name} FAILED ON CLOUD!")
                    print("--- BEGIN CLOUD LOG ---")
                    print(error_details[-1500:]) 
                    print("--- END CLOUD LOG ---\n")
                except requests.exceptions.RequestException:
                    print(f"\n❌ {job_name} FAILED ON CLOUD! (Could not fetch logs due to network error)")
                
                raise Exception(f"{job_name} job failed on the cloud platform.")
                
            print(f"  ...{job_name} processing, waiting 3 seconds...")
            time.sleep(3)
            
        # Catch momentary internet drops or DNS failures and just try again
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ Network glitch while polling: {str(e)[:50]}... Retrying in 3s...")
            time.sleep(3)
            continue
# ── 1. DATA INGESTION POINT FROM ESP32 ─────────────────────────────────────────
@app.route('/upload-data', methods=['POST'])
def handle_esp32_data():
    esp_payload = request.get_json()
    if not esp_payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    target_label = request.args.get('label', esp_payload.get('label', 'UNKNOWN')).upper()
    file_name = f"{target_label}_{int(time.time())}.json"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
        "x-label": target_label,
        "x-file-name": file_name
    }

    try:
        ei_response = requests.post(EI_INGESTION_URL, json=esp_payload, headers=headers)
        
        if ei_response.status_code in [200, 201]:
            if target_label in EXPECTED_CLASSES:
                uploaded_classes_registry.add(target_label)
                # Reset pipeline state when new data arrives (allows re-training)
                pipeline_state["complete"] = False
                pipeline_state["error"] = None
            
            missing_classes = EXPECTED_CLASSES - uploaded_classes_registry
            return jsonify({
                "status": "Success",
                "registered_class": target_label,
                "classes_received": list(uploaded_classes_registry),
                "ready_for_training": len(missing_classes) == 0,
                "missing_classes": list(missing_classes)
            }), 200
        else:
            return jsonify({
                "error": "Edge Impulse rejected data", 
                "status_code": ei_response.status_code,
                "details": ei_response.json() if ei_response.text else "No details"
            }), ei_response.status_code
            
    except Exception as e:
        return jsonify({"error": "Failed to proxy data", "details": str(e)}), 500

# ── 2. STATUS ENDPOINT FOR ESP32 POLLING ─────────────────────────────────────
@app.route('/status', methods=['GET'])
def get_status():
    missing_classes = EXPECTED_CLASSES - uploaded_classes_registry
    model_exists = os.path.exists(os.path.join(STATIC_MODEL_DIR, "model.tflite"))
    
    return jsonify({
        "classes_uploaded": list(uploaded_classes_registry),
        "missing_classes": list(missing_classes),
        "ready_for_training": len(missing_classes) == 0,
        "building": pipeline_state["building"],
        "build_complete": pipeline_state["complete"],
        "model_available": model_exists,
        "error": pipeline_state["error"],
        "message": pipeline_state["message"]
    }), 200

# ── 3. FULL AUTOMATED PIPELINE (FEATURES -> TRAIN -> BUILD -> EXTRACT) ─────────
@app.route('/build-and-extract', methods=['POST'])
def run_correct_server_pipeline():
    global pipeline_state
    
    # Prevent multiple simultaneous builds
    if pipeline_state["building"]:
        return jsonify({"status": "Building", "message": "Pipeline already in progress"}), 200
    
    if pipeline_state["complete"]:
        return jsonify({"status": "Complete", "message": "Model ready for download"}), 200
    
    missing_classes = EXPECTED_CLASSES - uploaded_classes_registry
    if missing_classes:
        return jsonify({"error": "Cannot train unified model yet. Missing class data.", "missing_classes": list(missing_classes)}), 400

    # Mark pipeline as building
    pipeline_state = {"building": True, "complete": False, "error": None, "message": "Pipeline started"}

    # Base headers for standard JSON endpoints
    json_headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

    try:
        # ── STEP 0: Fetch dynamic dspId and learnId ──
        print("\n🔍 Step 0: Fetching pipeline block IDs...")
        impulse_res = requests.get(f"{EI_STUDIO_URL}/api/{PROJECT_ID}/impulse", headers={"x-api-key": API_KEY})
        if impulse_res.status_code != 200:
            return jsonify({"error": "Failed to fetch impulse configuration", "details": impulse_res.text}), 500
            
        impulse_data = impulse_res.json().get("impulse", {})
        dsp_blocks = impulse_data.get("dspBlocks", [])
        learn_blocks = impulse_data.get("learnBlocks", [])
        
        if len(dsp_blocks) == 0 or len(learn_blocks) == 0:
            return jsonify({"error": "Impulse blocks not found. Save your impulse in Studio first."}), 500

        dsp_id = dsp_blocks[0].get("id")
        learn_id = learn_blocks[0].get("id")
        print(f"  ✅ Blocks found! DSP ID: {dsp_id} | Learn ID: {learn_id}")

        # ── STEP 1: Generate Features (DSP) ──
        print("⚙️ Step 1: Generating DSP Features...")
        features_res = requests.post(
            f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/generate-features", 
            json={"dspId": dsp_id}, headers=json_headers
        )
        if features_res.status_code != 200 or not features_res.json().get("success"):
            print(f"❌ DSP API Error: {features_res.text}")
            return jsonify({"error": "Failed to start features job", "details": features_res.text}), 500
            
        wait_for_job(features_res.json().get("id"), "Feature Generation")

        # ── STEP 2: Train the Keras Model (Correct Two-Step API Pattern) ──
        print("🧠 Step 2: Configuring and Triggering Training...")

        # 1. PRE-CONFIGURE: Set the architecture/hyperparameters first
        # This prevents the "Not updated configuration" error
        config_payload = {
            "mode": "visual",
            "trainingCycles": 50,
            "learningRate": 0.0005,
            "batchSize": 16,
            "trainTestSplit": 0.8,
            # Force this exact architecture
            "visualLayers": [
                {"type": "dense", "neurons": 16},
                {"type": "dense", "neurons": 8}
            ]
        }
        
        config_res = requests.post(
            f"{EI_STUDIO_URL}/api/{PROJECT_ID}/training/keras/{learn_id}",
            headers=json_headers,
            json=config_payload
        )
        
        if config_res.status_code != 200:
            print(f"❌ Configuration API Error: {config_res.text}")
            return jsonify({"error": "Failed to set training config", "details": config_res.text}), 500

        # 2. TRIGGER: Start the job using the configuration just saved
        # By sending the exact same payload, we ensure consistency.
        train_res = requests.post(
            f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/train/keras/{learn_id}",
            headers=json_headers,
            json=config_payload
        )
        
        if train_res.status_code != 200:
            # Check if this is an "already running" error
            try:
                error_data = train_res.json()
                if "already running" in error_data.get("error", "").lower():
                    print("⚠️ Training job already running, waiting for it to complete...")
                    # Extract the job ID from the error message
                    import re
                    job_id_match = re.search(r'job ID: (\d+)', error_data.get("error", ""))
                    if job_id_match:
                        existing_job_id = int(job_id_match.group(1))
                        wait_for_job(existing_job_id, "Existing Training Job")
                    else:
                        # If we can't extract ID, just wait a bit and continue
                        time.sleep(30)
            except:
                print(f"❌ Training API Error: {train_res.text}")
                return jsonify({"error": "Failed to trigger training job", "details": train_res.text}), 500
        else:
            if not train_res.json().get("success"):
                print(f"❌ Training API Error: {train_res.text}")
                return jsonify({"error": "Failed to trigger training job", "details": train_res.text}), 500
            wait_for_job(train_res.json().get("id"), "Neural Network Training")

        # ── STEP 3: Build the Deployment ZIP ──
        print("🚀 Step 3: Compiling Edge Device Model (Standard TFLite)...")
        
        # CRITICAL FIX: "engine": "tflite" disables the EON Compiler.
        # This preserves the model schema as a C++ hex array instead of stripping it.
        build_res = requests.post(
            f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/build-ondevice-model?type=zip", 
            json={"engine": "tflite"}, headers=json_headers
        )
        if build_res.status_code != 200 or not build_res.json().get("success"):
            return jsonify({"error": "Failed to start build job", "details": build_res.text}), 500
        wait_for_job(build_res.json().get("id"), "TFLite Build")

        # ── STEP 4: Download & Extract Raw TFLite Binary ──
        print("📥 Step 4: Downloading and parsing unified model...")
        download_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/deployment/download?type=zip"
        
        # Download endpoint only requires the API key header
        download_res = requests.get(download_url, headers={"x-api-key": API_KEY}, stream=True)
        
        if download_res.status_code != 200:
            return jsonify({"error": "Failed to download ZIP package"}), 400

        zip_bytes = io.BytesIO(download_res.content)
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            
            # 1. Look for a literal .tflite file first (just in case)
            tflite_filename = next((f.filename for f in zip_ref.infolist() if f.filename.endswith('.tflite')), None)
            
            if tflite_filename:
                raw_tflite_data = zip_ref.read(tflite_filename)
            else:
                # 2. If it's a C++ byte array, Python will extract and compile it back to binary!
                print("  ...Extracting hex array from C++ wrapper...")
                import re
                
                # Locate the C++ file containing the model array
                cpp_file = next((f.filename for f in zip_ref.infolist() if f.filename.endswith(('.cpp', '.h')) and 'tflite_learn_' in f.filename), None)
                if not cpp_file:
                    raise Exception("Could not find .tflite file or C++ model array in archive.")
                
                cpp_content = zip_ref.read(cpp_file).decode('utf-8')
                
                # Regex to grab everything between the array brackets: { 0x1c, 0x00... };
                match = re.search(r'=\s*\{([\s\S]*?)\}\;', cpp_content)
                if not match:
                    raise Exception("Failed to parse hex array from C++ file.")
                
                # Clean the string, split by commas, and convert hex values to pure bytes
                hex_strings = match.group(1).replace('\n', '').replace('\r', '').split(',')
                byte_list = [int(x.strip(), 16) for x in hex_strings if x.strip()]
                raw_tflite_data = bytes(byte_list)

            # Save the pure binary file for the ESP32 to download
            target_path = os.path.join(STATIC_MODEL_DIR, "model.tflite")
            with open(target_path, "wb") as f:
                f.write(raw_tflite_data)

        uploaded_classes_registry.clear()
        pipeline_state = {"building": False, "complete": True, "error": None, "message": "Model ready for download"}
        print("🎉 Pipeline complete! model.tflite is perfectly hot-swapped and ready for the ESP32.")
        return jsonify({"status": "Success", "message": "Autonomous ML pipeline complete. Model ready."}), 200
    except Exception as e:
        import traceback
        pipeline_state = {"building": False, "complete": False, "error": str(e), "message": "Pipeline failed"}
        print(f"\n❌ FATAL PIPELINE CRASH: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Pipeline failure", "details": str(e)}), 500
    
# ── 4. STATIC FILE ENDPOINT FOR ESP32 ──────────────────────────────────────────
@app.route('/models/model.tflite', methods=['GET'])
def serve_model_to_esp32():
    return send_from_directory(STATIC_MODEL_DIR, "model.tflite", as_attachment=True)

if __name__ == '__main__':
    # use_reloader=False prevents the pip_system_certs crash loop on Windows
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)