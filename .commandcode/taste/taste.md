# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# code-style
- When modifying existing code, do not add new features or functionality that weren't present in the original file — keep changes minimal and focused on fixing the specific issue. Confidence: 0.70

# Debugging
- Before suggesting code changes to fix a bug, first explain the root cause of why the issue is happening. Confidence: 0.90

# esp-idf
See [esp-idf/taste.md](esp-idf/taste.md)
# Logging
- For the ESP32 federated learning client, keep logging concise — only print essential status messages (connecting to server, sending/receiving weights, waiting for turn, training progress) and avoid verbose debug output. Confidence: 0.70

# communication-style
- When asked how to build/use a project that has documented code, first read the file headers and comments for existing instructions, then give a concise answer directly rather than verbose generic multi-step explanations. Confidence: 0.65

# Architecture
- For ESP32/ESP8266 federated learning code, separate into modular files: main file (networking/state/loop), training file, and inference file, to allow swapping models or approaches independently. Confidence: 0.85
- For TFLite Micro on ESP32, use ESP-IDF with PlatformIO and Espressif's official esp-tflite-micro component — Arduino IDE libraries (TensorFlowLite_ESP32, Adafruit_TensorFlow_Lite) have compatibility issues with modern ESP32 board packages. Confidence: 0.85
- When porting Arduino ESP32 code to ESP-IDF, port the complete functionality exactly (including HTTP communication, registration, polling) — do not leave TODOs or suggest simplified alternatives. Confidence: 0.80
- For the ESP32 federated learning client, use full TFLite Micro inference — do not suggest simplified manual inference as an alternative. Confidence: 0.85

