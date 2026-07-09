"""
Edge Impulse Local Server Middleware

Receives sensor data from ESP32 and forwards to Edge Impulse.
Polls Edge Impulse for new models, downloads and unzips them.
Serves TFLite models to ESP32.

Usage:
    python ei_local_server.py

Endpoints:
    POST /upload_data     — Receive sensor data from ESP32, forward to EI
    GET  /status          — Return current phase (train/inference/wait)
    GET  /get_model       — Serve TFLite model to ESP32
    POST /poll_ei_model   — Check EI for new model, download+unzip
    GET  /dashboard       — System status
"""

import os
import json
import time
import hmac
import hashlib
import zipfile
import requests
from flask import Flask, request, jsonify, send_file
from collections import defaultdict

app = Flask(__name__)

# ── Edge Impulse Config ────────────────────────────────────────────────────────
EI_API_KEY = "ei_fd83..."                    # Your EI project API key
EI_HMAC_KEY = "fed53116..."                  # Your EI HMAC signing key
EI_INGESTION_URL = "https://ingestion.edgeimpulse.com/api/training/data"
EI_DEPLOYMENT_URL = "https://studio.edgeimpulse.com/api/v1/deployment/model"

# ── Model storage ──────────────────────────────────────────────────────────────
MODEL_DIR = "ei_models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, "model.zip")

# ── FL State ───────────────────────────────────────────────────────────────────
MIN_CLIENTS = 2
UPLOAD_TIMEOUT_S = 300  # 5 minutes to collect all data

class ServerState:
    def __init__(self):
        self.phase = "waiting_for_clients"  # waiting, collecting, inference
        self.registered_clients = {}
        self.cycle_number = 0
        self.last_upload_time = {}
        self.model_version = 0
        self.model_path = None
        
    def reset(self):
        self.phase = "waiting_for_clients"
        self.last_upload_time = {}

state = ServerState()

# ── Ensure model directory exists ──────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

# ── HMAC Signing ───────────────────────────────────────────────────────────────
def sign_payload(data):
    """Sign EI payload with HMAC-SHA256."""
    data["signature"] = "0" * 64
    
    encoded = json.dumps(data)
    
    signature = hmac.new(
        bytes(EI_HMAC_KEY, 'utf-8'),
        msg=encoded.encode('utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    data["signature"] = signature
    return json.dumps(data)

# ── Forward data to Edge Impulse ───────────────────────────────────────────────
def forward_to_ei(payload, label, device_name):
    """Forward sensor data to Edge Impulse ingestion API."""
    # Build EI payload
    ei_data = {
        "protected": {
            "ver": "v1",
            "alg": "HS256",
            "iat": int(time.time())
        },
        "signature": "0" * 64,
        "payload": {
            "device_name": device_name,
            "device_type": "ESP32-MPU6050",
            "interval_ms": payload.get("interval_ms", 200),
            "sensors": payload.get("sensors", []),
            "values": payload.get("values", [])
        }
    }
    
    # Sign payload
    signed_data = sign_payload(ei_data)
    
    # Forward to EI
    headers = {
        "Content-Type": "application/json",
        "x-api-key": EI_API_KEY,
        "x-label": label,
        "x-file-name": f"{device_name}_{label}_{int(time.time())}"
    }
    
    try:
        response = requests.post(
            EI_INGESTION_URL,
            data=signed_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"[EI] Uploaded {label} data for {device_name}")
            return True
        else:
            print(f"[EI] Upload failed: {response.status_code} — {response.text}")
            return False
    except Exception as e:
        print(f"[EI] Upload error: {e}")
        return False

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.route("/upload_data", methods=["POST"])
def upload_data():
    """Receive sensor data from ESP32 and forward to Edge Impulse."""
    data = request.get_json(force=True)
    
    client_id = data.get("client_id", "unknown")
    label = data.get("label", "unknown")
    interval_ms = data.get("interval_ms", 200)
    sensors = data.get("sensors", [])
    values = data.get("values", [])
    
    if not values:
        return jsonify({"error": "no data"}), 400
    
    print(f"[Server] Received {len(values)} samples from {client_id} (label={label})")
    
    # Register client
    state.registered_clients[client_id] = time.time()
    state.last_upload_time[client_id] = time.time()
    
    # Forward to Edge Impulse
    success = forward_to_ei(
        {"interval_ms": interval_ms, "sensors": sensors, "values": values},
        label,
        client_id
    )
    
    if success:
        return jsonify({"status": "uploaded", "samples": len(values)})
    else:
        return jsonify({"error": "EI upload failed"}), 500

@app.route("/status", methods=["GET"])
def status():
    """Return current phase to ESP32."""
    client_id = request.args.get("client_id", "unknown")
    
    # Register client on poll
    state.registered_clients[client_id] = time.time()
    
    n_clients = len(state.registered_clients)
    
    response = {
        "phase": state.phase,
        "cycle": state.cycle_number,
        "registered_clients": n_clients,
    }
    
    if state.phase == "waiting_for_clients":
        if n_clients >= MIN_CLIENTS:
            state.phase = "collecting"
            response["phase"] = "collecting"
            response["instruction"] = "train"
        else:
            response["instruction"] = "wait"
            response["waiting_for"] = MIN_CLIENTS - n_clients
    
    elif state.phase == "collecting":
        # Check if all clients have uploaded recently
        all_uploaded = True
        for cid in state.registered_clients:
            last = state.last_upload_time.get(cid, 0)
            if time.time() - last > UPLOAD_TIMEOUT_S:
                all_uploaded = False
                break
        
        if all_uploaded and len(state.last_upload_time) >= MIN_CLIENTS:
            state.phase = "inference"
            response["phase"] = "inference"
            response["instruction"] = "inference"
        else:
            response["instruction"] = "train"
    
    elif state.phase == "inference":
        response["instruction"] = "inference"
    
    return jsonify(response)

@app.route("/get_model", methods=["GET"])
def get_model():
    """Serve TFLite model to ESP32."""
    if not state.model_path or not os.path.exists(state.model_path):
        return jsonify({"error": "No model available"}), 404
    
    return send_file(
        state.model_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='model.tflite'
    )

@app.route("/poll_ei_model", methods=["POST"])
def poll_ei_model():
    """Check Edge Impulse for new model, download and unzip."""
    print("[Server] Polling Edge Impulse for new model...")
    
    headers = {
        "x-api-key": EI_API_KEY
    }
    
    try:
        # Check for model
        response = requests.get(
            EI_DEPLOYMENT_URL,
            headers=headers,
            params={"type": "tflite", "variant": "float32"},
            timeout=30
        )
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            
            if "zip" in content_type or response.content[:2] == b'PK':
                # Save zip
                with open(MODEL_ZIP_PATH, 'wb') as f:
                    f.write(response.content)
                print(f"[Server] Downloaded model zip ({len(response.content)} bytes)")
                
                # Unzip
                with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                    # Find .tflite file in zip
                    tflite_files = [f for f in zip_ref.namelist() if f.endswith('.tflite')]
                    if tflite_files:
                        zip_ref.extract(tflite_files[0], MODEL_DIR)
                        state.model_path = os.path.join(MODEL_DIR, tflite_files[0])
                        state.model_version += 1
                        print(f"[Server] Extracted {tflite_files[0]} (version {state.model_version})")
                    else:
                        print("[Server] No .tflite file found in zip")
                        return jsonify({"error": "No .tflite in zip"}), 500
            
            elif "tflite" in content_type or response.content[:4] == b'\x1c\x00\x00\x00':
                # Direct TFLite file
                state.model_path = MODEL_PATH
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.content)
                state.model_version += 1
                print(f"[Server] Downloaded TFLite model ({len(response.content)} bytes, version {state.model_version})")
            
            else:
                print(f"[Server] Unknown content type: {content_type}")
                return jsonify({"error": f"Unknown content type: {content_type}"}), 500
            
            return jsonify({
                "status": "model_updated",
                "version": state.model_version,
                "path": state.model_path
            })
        
        elif response.status_code == 404:
            print("[Server] No model available yet")
            return jsonify({"status": "no_model"}), 404
        
        else:
            print(f"[Server] EI API error: {response.status_code}")
            return jsonify({"error": f"EI API error: {response.status_code}"}), 500
    
    except Exception as e:
        print(f"[Server] Poll error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """System status."""
    return jsonify({
        "phase": state.phase,
        "cycle": state.cycle_number,
        "registered_clients": list(state.registered_clients.keys()),
        "model_version": state.model_version,
        "model_path": state.model_path,
        "last_uploads": {k: time.time() - v for k, v in state.last_upload_time.items()},
    })

@app.route("/reset", methods=["POST"])
def reset():
    """Reset server state for new cycle."""
    state.cycle_number += 1
    state.phase = "waiting_for_clients"
    state.last_upload_time = {}
    print(f"[Server] Reset for cycle {state.cycle_number}")
    return jsonify({"status": "reset", "cycle": state.cycle_number})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Edge Impulse Local Server Middleware")
    print(f"  Ingestion URL: {EI_INGESTION_URL}")
    print(f"  Deployment URL: {EI_DEPLOYMENT_URL}")
    print(f"  Model dir: {MODEL_DIR}")
    print(f"  Min clients: {MIN_CLIENTS}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
