import os
import io
import time
import zipfile
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
EI_STUDIO_URL = "https://studio.edgeimpulse.com/v1"
EI_INGESTION_URL = "https://ingestion.edgeimpulse.com/api/training/data"

PROJECT_ID = 956111  # Update with your Edge Impulse Project ID
API_KEY = "ei_a81196b2e12fc6b901edaf7fc05077eabebb9d6ea5026aaf"  # Update with your Project API Key

STATIC_MODEL_DIR = "./static_bin"
os.makedirs(STATIC_MODEL_DIR, exist_ok=True)

EXPECTED_CLASSES = {"STILL", "WALKING", "SHAKING", "TAPPING"}
uploaded_classes_registry = set()

# ── 1. DATA INGESTION POINT FROM ESP32 ─────────────────────────────────────────
@app.route('/upload-data', methods=['POST'])
def handle_esp32_data():
    esp_payload = request.get_json()
    if not esp_payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    target_label = request.args.get('label', esp_payload.get('label', 'UNKNOWN')).upper()
    
    # Build EI Data Acquisition Format
    ei_payload = {
        "protected": {
            "ver": "v1",
            "alg": "none",
            "iat": int(time.time())
        },
        "signature": "00",
        "payload": esp_payload.get("payload", esp_payload)
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
        "x-label": target_label,
        "x-file-name": f"esp32_{target_label}_{int(time.time())}.json"
    }

    try:
        ei_response = requests.post(EI_INGESTION_URL, json=ei_payload, headers=headers)
        
        if ei_response.status_code in [200, 201]:
            if target_label in EXPECTED_CLASSES:
                uploaded_classes_registry.add(target_label)
            
            missing_classes = EXPECTED_CLASSES - uploaded_classes_registry
            return jsonify({
                "status": "Success",
                "registered_class": target_label,
                "classes_received": list(uploaded_classes_registry),
                "ready_for_training": len(missing_classes) == 0,
                "missing_classes": list(missing_classes)
            }), 200
        else:
            return jsonify({"error": "Edge Impulse rejected data", "details": ei_response.text}), ei_response.status_code
            
    except Exception as e:
        return jsonify({"error": "Failed to proxy data", "details": str(e)}), 500

# ── 2. CORRECT SERVER PIPELINE (BUILD -> POLL -> DOWNLOAD -> EXTRACT) ──────────
@app.route('/build-and-extract', methods=['POST'])
def run_correct_server_pipeline():
    missing_classes = EXPECTED_CLASSES - uploaded_classes_registry
    if missing_classes:
        return jsonify({
            "error": "Cannot build unified model yet. Missing class data.",
            "missing_classes": list(missing_classes)
        }), 400

    headers = {"x-api-key": API_KEY}

    try:
        # STEP 1: Trigger a Build Job
        build_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/build-ondevice-model?type=zip"
        build_payload = {"engine": "tflite-eon"}
        
        print("🚀 Step 1: Triggering compilation build job on Edge Impulse...")
        build_res = requests.post(build_url, json=build_payload, headers=headers)
        if build_res.status_code != 200 or not build_res.json().get("success"):
            return jsonify({"error": "Failed to trigger build job", "details": build_res.json()}), 500
        
        job_id = build_res.json().get("id")
        print(f"  Job triggered successfully. Job ID: {job_id}")

        # STEP 2: Poll Until Job Completes
        status_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/jobs/{job_id}/status"
        print("⏳ Step 2: Polling compilation job status...")
        
        while True:
            status_res = requests.get(status_url, headers=headers)
            if status_res.status_code != 200:
                return jsonify({"error": "Failed to poll job status"}), 500
                
            job_status = status_res.json().get("job", {})
            
            if job_status.get("finishedSuccessful"):
                print("  ✅ Job finished successfully!")
                break
            elif job_status.get("finished"):
                return jsonify({"error": "Edge Impulse build job failed on the cloud platform"}), 500
                
            print("  ...Job still processing, waiting 2 seconds...")
            time.sleep(2)

        # STEP 3: Download the ZIP file
        download_url = f"{EI_STUDIO_URL}/api/{PROJECT_ID}/deployment/download?type=zip"
        print("📥 Step 3: Downloading unified ZIP deployment package...")
        download_res = requests.get(download_url, headers=headers, stream=True)
        
        if download_res.status_code != 200:
            return jsonify({"error": "Failed to download model package binary"}), 400

        # STEP 4: Server Extracts TFLite & Serves to ESP32
        print("📦 Step 4: Extracting flat .tflite model from deployment archive...")
        zip_bytes = io.BytesIO(download_res.content)
        
        with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
            tflite_filename = None
            # Find the file within the 'tflite-model/' path structure
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.tflite'):
                    tflite_filename = file_info.filename
                    break
            
            if not tflite_filename:
                return jsonify({"error": "No raw .tflite file found in the generated ZIP"}), 500
            
            print(f"  Isolating raw model binary block: {tflite_filename}")
            raw_tflite_data = zip_ref.read(tflite_filename)
            
            # Save raw model binary to static serve endpoint file location
            target_path = os.path.join(STATIC_MODEL_DIR, "model.tflite")
            with open(target_path, "wb") as f:
                f.write(raw_tflite_data)

        # Reset registry tracker for the next iteration loop
        uploaded_classes_registry.clear()
        return jsonify({"status": "Success", "message": "Unified multi-class model hot-swapped and ready for ESP32 fetch."}), 200

    except Exception as e:
        return jsonify({"error": "Pipeline exception occurred", "details": str(e)}), 500

# ── 3. STATIC FILE ENDPOINT FOR ESP32 ──────────────────────────────────────────
@app.route('/models/model.tflite', methods=['GET'])
def serve_model_to_esp32():
    return send_from_directory(STATIC_MODEL_DIR, "model.tflite", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)