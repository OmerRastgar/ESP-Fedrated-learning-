import time
import random
import requests

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SERVER_IP = "127.0.0.1"  # Change to your server's IP if running remotely
SERVER_PORT = 5000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}"

INPUT_DIM = 9
LOCAL_SAMPLES_PER_CLASS = 40
CLASS_NAMES = ["STILL", "WALKING", "SHAKING", "TAPPING"]

def generate_dummy_sensor_data(class_name):
    """
    Generates a mock payload structure containing 40 samples of 9D sensor vectors.
    Slightly alters values depending on the class to mimic real physical variance.
    """
    samples = []
    for _ in range(LOCAL_SAMPLES_PER_CLASS):
        if class_name == "STILL":
            # Mostly resting gravity on Z-axis, minimal gyro noise
            sample = [0.0, 0.0, 9.81] + [random.uniform(-0.01, 0.01) for _ in range(6)]
        elif class_name == "WALKING":
            # Cyclic rhythmic dynamic motion acceleration
            sample = [random.uniform(-2.0, 2.0), random.uniform(8.0, 11.0), random.uniform(-1.0, 3.0)] + [random.uniform(-0.5, 0.5) for _ in range(6)]
        elif class_name == "SHAKING":
            # High frequency, high amplitude spikes across all axes
            sample = [random.uniform(-15.0, 15.0) for _ in range(3)] + [random.uniform(-5.0, 5.0) for _ in range(6)]
        elif class_name == "TAPPING":
            # Sharp isolated impulse shocks
            sample = [0.0, 0.0, 9.81 + random.choice([0.0, 12.0])] + [random.uniform(-0.1, 0.1) for _ in range(6)]
        else:
            sample = [random.uniform(-1.0, 1.0) for _ in range(INPUT_DIM)]
            
        samples.append(sample)
        
    # Format payload structural block matching the expected Edge Impulse structure
    payload = {
        "protected": {
            "ver": "v1",
            "alg": "none",
            "apiKey": "placeholder_handled_by_server"
        },
        "signature": "empty",
        "payload": {
            "device_name": "simulated-esp32-node",
            "device_type": "ESP32",
            "interval_ms": 200,
            "sensors": [
                {"name": "ax", "units": "m/s2"}, {"name": "ay", "units": "m/s2"}, {"name": "az", "units": "m/s2"},
                {"name": "gx", "units": "deg/s"}, {"name": "gy", "units": "deg/s"}, {"name": "gz", "units": "deg/s"},
                {"name": "mx", "units": "uT"}, {"name": "my", "units": "uT"}, {"name": "mz", "units": "uT"}
            ],
            "values": samples
        }
    }
    return payload

def run_simulation():
    print("🚀 Starting Middleman Server Integration Test...\n")
    
    # ── STEP 1: SEQUENTIAL CLASS DATA UPLOADS ─────────────────────────────────
    for cls_name in CLASS_NAMES:
        print(f"📦 Simulating collection for class: {cls_name}")
        payload = generate_dummy_sensor_data(cls_name)
        
        # Post directly to the server's data endpoint with the target label parameter
        upload_url = f"{BASE_URL}/upload-data?label={cls_name}"
        try:
            response = requests.post(upload_url, json=payload)
            if response.status_code == 200:
                print(f"  ✅ Server Response: {response.json().get('registered_class')} successfully ingested.")
                print(f"  📊 Missing classes remaining: {response.json().get('missing_classes')}\n")
            else:
                print(f"  ❌ Upload failed with status code {response.status_code}: {response.text}")
                return
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection Error: Is your middleman server running on {BASE_URL}?")
            return
            
        time.sleep(1) # Short breathing window between sequential uploads
        
    # ── STEP 2: TRIGGER BUNDLED BUILD ENGINE ─────────────────────────────────
    print("🔄 All classes submitted! Requesting Edge Impulse to compile a combined model...")
    build_url = f"{BASE_URL}/build-and-extract"
    
    try:
        build_response = requests.post(build_url)
        if build_response.status_code == 200:
            print(f"  ✅ Build Successful: {build_response.json().get('status')}\n")
        else:
            print(f"  ❌ Build failed: {build_response.json().get('error')}")
            print(f"  Details: {build_response.json().get('missing_classes')}")
            return
    except Exception as e:
        print(f"  ❌ Error during build process: {str(e)}")
        return

    # ── STEP 3: DOWNLOAD AND STORE EXTRACTED RAW .TFLITE BINARY ──────────────
    print("📥 Downloading the raw extracted model from the server endpoint...")
    model_url = f"{BASE_URL}/models/model.tflite"
    
    try:
        model_response = requests.get(model_url, stream=True)
        if model_response.status_code == 200:
            output_filename = "downloaded_test_model.tflite"
            with open(output_filename, "wb") as f:
                f.write(model_response.content)
            print(f"  🎉 Test complete! Raw binary model saved locally as: '{output_filename}'")
        else:
            print(f"  ❌ Failed to download model file from static endpoint. Status: {model_response.status_code}")
    except Exception as e:
        print(f"  ❌ Download execution failed: {str(e)}")

if __name__ == "__main__":
    run_simulation()