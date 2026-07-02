# Federated Learning with ESP32 & TFLite — Knowledge Base

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Key Learnings](#key-learnings)
4. [ESP-IDF vs Arduino](#esp-idf-vs-arduino)
5. [TFLite Integration](#tflite-integration)
6. [Common Issues & Solutions](#common-issues--solutions)
7. [Best Practices](#best-practices)
8. [Quick Reference](#quick-reference)

---

## Project Overview

**What we built:**
- Federated Learning system with ESP32 clients and Python server
- Sequential training: one client trains at a time, server aggregates via FedAvg
- Dual inference: Manual forward pass + TFLite Micro inference for comparison
- TFLite model generated on server after each FedAvg round, downloaded by ESP32

**Components:**
- `fl_server.py` — Flask server managing FL orchestration, FedAvg, TFLite model generation
- `tflite_model_manager.py` — Converts server weights to TFLite flatbuffer format
- `FL_client_esp32_tflite/` — ESP-IDF project with TFLite Micro inference
- `FL_client_esp32_modular/` — Arduino modular version (manual inference)
- `FL_client_esp32/` — Original Arduino monolithic version

---

## Architecture

### Model Architecture
```
Input (9) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(4, Softmax)
```
- **9 inputs:** Accelerometer XYZ + Gyroscope XYZ + 3 reserved zeros
- **4 outputs:** STILL(0), WALKING(1), SHAKING(2), TAPPING(3)
- **332 total weights:** W1(9×16) + b1(16) + W2(16×8) + b2(8) + W3(8×4) + b3(4)

### FL Flow
```
1. ESP32 registers with server
2. Server waits for MIN_CLIENTS (2) to register
3. Server starts Round 1, instructs first client to train
4. Client trains → uploads weights → next client trains
5. After all clients: server runs FedAvg
6. Server updates TFLite model from global weights
7. Repeat until convergence or MAX_ROUNDS
8. Enter inference mode: ESP32 downloads TFLite model, runs both inferences
```

### Weight Format Mismatch (Critical)
**ESP32 client sends:** Flat array of 332 floats
**Server expects:** Nested arrays `[W1, b1, W2, b2, W3, b3]` for FedAvg

**Fix in server `/update` endpoint:**
```python
# Reshape flat weights to nested format
flat_arr = np.array(flat, dtype=float)
shapes = [
    (INPUT_DIM, HIDDEN1), (HIDDEN1,),
    (HIDDEN1, HIDDEN2), (HIDDEN2,),
    (HIDDEN2, OUTPUT_DIM), (OUTPUT_DIM,),
]
reshaped_weights = []
idx = 0
for s in shapes:
    size = s[0] if len(s) == 1 else s[0] * s[1]
    reshaped_weights.append(flat_arr[idx:idx+size].reshape(s).tolist())
    idx += size
weights = reshaped_weights
```

---

## Key Learnings

### 1. Arduino TFLite Libraries Don't Work
- `TensorFlowLite_ESP32` by tanakamasayuki: Broken with modern ESP32 board packages
- `Adafruit_TensorFlow_Lite`: Depends on the same broken library
- **Solution:** Use ESP-IDF with Espressif's official `esp-tflite-micro` component

### 2. ESP-IDF is the Only Reliable Way for TFLite on ESP32
- Uses `idf.py` component manager to download `esp-tflite-micro`
- Managed components: `idf_component.yml` + `idf.py add-dependency`
- **Not supported by PlatformIO** (PlatformIO doesn't handle IDF managed components)

### 3. Manual Inference is Better for Small Models
- For 332 weights: Manual inference is ~0.2ms, TFLite is ~0.3ms
- TFLite overhead (interpreter, arena, managed ops) not justified for <5K weights
- Manual forward pass has no dependencies, simpler debugging
- **TFLite crossover point:** ~10K+ weights where quantization benefits emerge

### 4. TFLite Model is 2.3x Larger Than Raw Weights
- Raw weights: 332 floats × 4 bytes = 1,328 bytes
- TFLite model: ~3,000-3,500 bytes (includes flatbuffer metadata, op codes, tensor definitions)

### 5. Server Must Regenerate TFLite on Every Request
- Don't cache stale TFLite models
- Always call `update_tflite_model(state.global_weights)` before serving
- ESP32 downloads once on first inference after entering inference mode

---

## ESP-IDF vs Arduino

### ESP-IDF Advantages
- Official TFLite Micro support via `esp-tflite-micro`
- FreeRTOS tasks (dual-core support)
- Better memory management
- Professional SDK with full hardware control

### ESP-IDF Disadvantages
- Steeper learning curve
- No Arduino `String` class (use custom `SimpleString` or C strings)
- No `Serial.println()` (use `ESP_LOGI()` instead)
- Legacy I2C driver deprecated but still works
- `app_main()` must not return (add `while(1) vTaskDelay(...)` at end)

### Code Porting (Arduino → ESP-IDF)

| Arduino | ESP-IDF |
|---------|---------|
| `Serial.printf()` | `ESP_LOGI(TAG, ...)` |
| `String` | `SimpleString` or `char[]` + `snprintf()` |
| `Wire.begin(21, 22)` | `i2c_param_config()` + `i2c_driver_install()` |
| `Wire.beginTransmission()` | `i2c_cmd_link_create()` + `i2c_master_*()` |
| `HTTPClient http` | `esp_http_client_*()` |
| `http.begin(url)` | `esp_http_client_init(&config)` |
| `http.GET()` / `http.POST()` | `esp_http_client_open()` + `esp_http_client_read()` |
| `setup()` / `loop()` | `app_main()` with FreeRTOS task loop |
| `delay(ms)` | `vTaskDelay(pdMS_TO_TICKS(ms))` |
| `millis()` | `xTaskGetTickCount()` |

### I2C (Legacy Driver — still works in IDF v6)
```cpp
#include "driver/i2c.h"

void i2c_master_init() {
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t)21;
    conf.scl_io_num = (gpio_num_t)22;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = 100000;
    conf.clk_flags = 0;
    i2c_param_config(I2C_NUM_0, &conf);
    i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0);
}
```

### HTTP Client (Must use open/read/close, not perform)
```cpp
// WRONG — perform() consumes response body
esp_http_client_perform(client);
esp_http_client_read(client, response, max_len);  // Returns 0!

// CORRECT — open + read
esp_http_client_open(client, 0);
esp_http_client_fetch_headers(client);
esp_http_client_read(client, response, max_len - 1);
esp_http_client_close(client);
```

---

## TFLite Integration

### ESP32 Side
```cpp
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TFLITE_ARENA_SIZE (32 * 1024)
static uint8_t tensor_arena[TFLITE_ARENA_SIZE] __attribute__((aligned(32)));

// Use MicroMutableOpResolver (not AllOpsResolver — doesn't exist in newer versions)
static tflite::MicroMutableOpResolver<10> resolver;
resolver.AddFullyConnected();
resolver.AddRelu();
resolver.AddSoftmax();
resolver.AddReshape();
```

### Server Side
```python
# Generate TFLite model from weights
def update_tflite_model(weights_list):
    model = weights_to_keras_model(weights_list)  # Set weights in Keras
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("fl_model.tflite", 'wb') as f:
        f.write(tflite_model)
```

### TFLite Model Validation
- TFLite files start with `0x1c` (flatbuffer magic byte)
- Minimum valid size: ~2,000 bytes for a small model
- Check `first_byte == 0x1c` and `size > 2000` before loading

---

## Common Issues & Solutions

### 1. ESP32 Keeps Rebooting (Guru Meditation Error)
**Cause:** TFLite model too small or corrupted (server returned error JSON, not binary)
**Fix:** Validate model size and magic byte before loading

### 2. All Clients Get "Wait" Forever
**Cause:** Both ESP32s flashed with same `CLIENT_ID`
**Fix:** Each device must have unique `CLIENT_ID`

### 3. 409 Error on Upload
**Cause:** Client uploading out of turn (server already moved to next client/round)
**Fix:** Sequential training design — client must wait for server's "train" instruction

### 4. TFLite Always Predicts Same Class
**Cause:** TFLite model stale (downloaded before FedAvg completed)
**Fix:** Server must regenerate TFLite on every `/download_tflite` request

### 5. Manual vs TFLite Show Different Values
**Cause:** TFLite model from server, manual weights from `/status` — different sources
**Fix:** Both should use same converged weights (server regenerates TFLite after FedAvg)

### 6. Untrained Classes Show High Probability
**Cause:** All 4 output biases start at 0, untrained classes have random weights
**Fix:** Either train all 4 classes, or set untrained biases to -100 in `unflattenWeights()`

### 7. GPIO Conflict on ESP-IDF v6
**Cause:** New I2C driver (`i2c_master.h`) has pin conflict detection
**Fix:** Use legacy driver (`driver/i2c.h`) with deprecation warning suppressed

### 8. `app_main()` Returns Immediately
**Cause:** ESP-IDF tasks end when `app_main()` returns (unlike Arduino's `loop()`)
**Fix:** Add `while(1) { vTaskDelay(pdMS_TO_TICKS(1000)); }` at end of `app_main()`

---

## Best Practices

### Modular Arduino Code
```
FL_client_esp32_modular/
├── FL_client_esp32_modular.ino  (main: WiFi, server, sensor, state machine)
├── model_config.h               (dimensions, weights, init, serialization)
├── inference.h                  (forward pass, prediction)
└── training.h                   (SGD, loss, data collection, training loop)
```

**To swap approaches:** Replace `inference.h` or `training.h` — main file unchanged

### ESP-IDF Project Structure
```
FL_client_esp32_tflite/
├── CMakeLists.txt
├── platformio.ini
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   └── idf_component.yml
├── managed_components/
│   └── espressif__esp-tflite-micro/
└── build/
```

### Log Level Control (ESP-IDF)
```cpp
// In app_main(), suppress all non-FL logs
esp_log_level_set("*", ESP_LOG_WARN);
esp_log_level_set("FL_CLIENT", ESP_LOG_INFO);
```

### Memory Management
- **Task stack:** 32KB for FL task (HTTP + TFLite)
- **TFLite arena:** 32KB aligned to 32 bytes
- **HTTP response buffer:** 4KB max
- **JSON body buffer:** 4KB max (use `SimpleString` with dynamic realloc)

---

## Quick Reference

### Build Commands
```powershell
# ESP-IDF build
idf.py build

# Flash
idf.py -p COM7 flash

# Monitor
idf.py -p COM7 monitor

# Flash + monitor
idf.py -p COM7 flash monitor

# Clean build
idf.py fullclean

# Add managed component
idf.py add-dependency "espressif/esp-tflite-micro"
```

### Arduino Build
```
Arduino IDE → Sketch → Upload
Serial Monitor: 115200 baud
```

### Server Commands
```powershell
# Start server
python fl_server.py

# Check dashboard
curl http://10.219.84.92:5000/dashboard

# Delete stale TFLite model
del tflite_models\fl_model.tflite
```

### ESP-IDF IDF PowerShell
```
Desktop shortcut: IDF_v6.0.2_Powershell.lnk
Or: powershell -Command ". 'C:\Espressif\tools\Microsoft.v6.0.2.PowerShell_profile.ps1'"
```

---

## File Locations

| File | Path |
|------|------|
| Server | `D:\Fedrated learning\fl_server.py` |
| TFLite Manager | `D:\Fedrated learning\tflite_model_manager.py` |
| ESP-IDF TFLite | `D:\Fedrated learning\FL_client_esp32_tflite\` |
| Arduino Modular | `D:\Fedrated learning\FL_client_esp32_modular\` |
| Arduino Original | `D:\Fedrated learning\FL_client_esp32\` |
| ESP8266 | `D:\Fedrated learning\FL_client_esp8266\` |
| TFLite Models | `D:\Fedrated learning\tflite_models\` |
| Taste File | `D:\Fedrated learning\.commandcode\taste\taste.md` |

---

*Last updated: 2026-07-01*
*Session: Federated Learning with ESP32 + TFLite Micro*
