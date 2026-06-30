# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# code-style
- When modifying existing code, do not add new features or functionality that weren't present in the original file — keep changes minimal and focused on fixing the specific issue. Confidence: 0.70

# Architecture
- For ESP32/ESP8266 federated learning code, separate into modular files: main file (networking/state/loop), training file, and inference file, to allow swapping models or approaches independently. Confidence: 0.85

