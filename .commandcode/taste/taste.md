# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# code-style
- Before writing code for a complex architectural change or significant refactor, first present a brief plan to the user for approval before executing. Confidence: 0.65
- When modifying existing code, do not add new features, functionality, or extra logging that weren't present in the original file — keep changes minimal and focused on fixing the specific issue. Confidence: 0.90
- When fixing a bug that has a root cause with a clear alternative, explain the root cause first before making the fix. Confidence: 0.92

# Debugging
- Before suggesting code changes to fix a bug, first explain the root cause of why the issue is happening. Confidence: 0.90

# esp-idf
See [esp-idf/taste.md](esp-idf/taste.md)
# Logging
- For the ESP32 federated learning client, keep logging concise — only print essential status messages (connecting to server, sending/receiving weights, waiting for turn, training progress) and avoid verbose debug output. Confidence: 0.70
- When logging stack or heap metrics for ESP32 tasks, show the usage as "used/total (% free)" so it's clear how much headroom remains (e.g., `Stack: 6116/8192 bytes (75% free)`). Confidence: 0.70
- Log when manual weights are successfully loaded from server status response and when TFLite model is initialized, to confirm both inference paths have fresh weights. Confidence: 0.65
- When refactoring or rewriting ESP32 federated learning code, preserve all existing benchmarking metrics (inference timing, heap usage, probabilities, training latency, HTTP latency, stack high water marks) — do not remove or simplify them during refactoring. Confidence: 0.75

# Training
- Use gradient clipping (cap gradients at ±1.0) in ESP32 client SGD training to prevent NaN/Inf weight explosion when training on converged weights from previous FL cycles. Confidence: 0.70
- Validate flattened weights for NaN/Inf before uploading from ESP32 client to FL server — skip upload if corrupted and log the error. Confidence: 0.75

# FL Server
- When receiving weight updates from ESP32 clients, validate that no individual weight value is NaN or Inf and reject with HTTP 400 if detected. Confidence: 0.75

# communication-style
- When asked how to build/use a project that has documented code, first read the file headers and comments for existing instructions, then give a concise answer directly rather than verbose generic multi-step explanations. Confidence: 0.65

# FL Server
- After cooldown ends in a federated learning cycle, keep the existing converged global weights and continue aggregation from there in the next cycle — do not reinitialize to random (Xavier) weights. Confidence: 0.85
- When the FL aggregation runs (FedAvg, convergence check), hold the server lock for the entire operation to prevent race conditions where a client poll triggers a new round before convergence is fully evaluated. Confidence: 0.80

# git
- Do not push individual fixes to the repository until all development/changes are fully complete and tested. Confidence: 0.90

# ESP-IDF Client
- When parsing weights from the FL server JSON `/status` response in `pollStatus()`, check the return value of `parseWeightsFromJson()` — if parsing fails, return `false` instead of silently using corrupted/partial weights. Confidence: 0.70
- When a new FL training cycle is detected during inference mode, reset `tflite_model_downloaded = false` and `tflite_initialized = false` so the client downloads the updated TFLite model (with new cycle's weights) before entering inference mode again. Confidence: 0.80

# Edge Impulse
- When sending data to Edge Impulse ingestion API, use `"alg": "none"` and `"signature": "00"` for the protected/signature fields, not HMAC-SHA256 signing. Confidence: 0.75
- When triggering Edge Impulse model training, use `POST /v1/api/{projectId}/jobs/build-ondevice-model?type=tflite` with body `{"engine": "tflite", "modelType": "int8"}` rather than the deployment download API. Confidence: 0.70
- Send the x-label header (not query parameter) when ingesting data to Edge Impulse. Confidence: 0.70

# Architecture
- For ESP32/ESP8266 federated learning code, separate into modular files: main file (networking/state/loop), training file, and inference file, to allow swapping models or approaches independently. Confidence: 0.85
- When using FreeRTOS tasks on ESP32, avoid large stack-allocated buffers (weight arrays, HTTP response buffers, JSON builders) — use static storage with `__attribute__((aligned(4)))` to prevent stack overflow (StoreProhibited) and alignment faults (LoadStoreAlignment). Confidence: 0.70
- For TFLite Micro on ESP32, use ESP-IDF with PlatformIO and Espressif's official esp-tflite-micro component — Arduino IDE libraries (TensorFlowLite_ESP32, Adafruit_TensorFlow_Lite) have compatibility issues with modern ESP32 board packages. Confidence: 0.85
- When porting Arduino ESP32 code to ESP-IDF, port the complete functionality exactly (including HTTP communication, registration, polling) — do not leave TODOs or suggest simplified alternatives. Confidence: 0.80
- For the ESP32 federated learning client, use full TFLite Micro inference — do not suggest or retain simplified manual inference as an alternative. Confidence: 0.88

