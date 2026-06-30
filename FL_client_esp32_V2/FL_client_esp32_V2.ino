/*
 * Federated Learning Client — ESP32 (Arduino IDE)
 *
 * Activity classifier: STILL(0) | WALKING(1) | SHAKING(2) | TAPPING(3)
 * Sensor: MPU6050 — 6 real features (ax,ay,az,gx,gy,gz) + 3 zeros = INPUT_DIM 9
 *
 * Key design decisions vs previous revision:
 *   - Output layer uses SOFTMAX (not linear) → probability distribution over 4 classes
 *   - Loss is CROSS-ENTROPY (not MSE) → correct gradient for classification
 *   - SGD gradient = softmax_out - one_hot_label  (softmax+CE combined derivative)
 *   - Training cycles all 4 classes locally each round (no server involvement):
 *       STILL → WALKING → SHAKING → TAPPING, each with LOCAL_SAMPLES_PER_CLASS
 *       samples. All classes are trained before weights are uploaded once.
 *   - Xavier init from ESP32 hardware RNG (esp_random) used as fallback if
 *     server unreachable on boot
 *   - Inference calls infer() which returns argmax class index + name string
 *
 * ── TinyML addition (inference-only, training is UNCHANGED) ──────────────────
 *   On FL convergence the server converts final weights to an int8 .tflite
 *   model. This client downloads it, verifies its SHA256 hash against the
 *   server before trusting it, and during the inference phase runs BOTH:
 *     (a) the existing hand-written float32 forward pass, and
 *     (b) a TensorFlow Lite Micro (TFLM) interpreter on the same sample
 *   Latency, memory footprint, and confidence are logged for both and
 *   periodically uploaded to the server as a benchmark report.
 *
 * Server-managed round lifecycle (clients poll /status):
 *   wait            → stay idle, poll again after POLL_INTERVAL_MS
 *   train           → download model, train all 4 classes, upload weights  [UNCHANGED]
 *   download_model  → download .tflite + verify hash before inference starts [NEW]
 *   inference       → run dual inference (hand-written + TFLM), benchmark
 *
 * Change WIFI_SSID, WIFI_PASS, SERVER_IP, CLIENT_ID before flashing each node.
 *
 * Library requirement: install "Chirale_TensorFlowLite" or the official
 * "TensorFlowLite_ESP32" library via Arduino Library Manager before compiling.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <math.h>
#include <mbedtls/sha256.h>   // for SHA256 hash verification of .tflite
#include "esp_random.h"       // ESP32 hardware RNG for Xavier fallback init

// ── TensorFlow Lite Micro (TFLM) — inference-only, no training APIs used ──────
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── Config ────────────────────────────────────────────────────────────────────
const char* WIFI_SSID   = "Cybergaar";
const char* WIFI_PASS   = "curedata4low";
const char* SERVER_IP   = "10.219.84.92";
const int   SERVER_PORT = 5000;
const char* CLIENT_ID             = "esp32-node-A";   // unique per device

const int   POLL_INTERVAL_MS          = 3000;  // idle poll cadence
const int   INFERENCE_INTERVAL_MS     = 500;   // forward pass cadence in inference mode
const int   SAMPLE_INTERVAL_MS        = 200;    // ms between MPU reads during collection
const int   LOCAL_SAMPLES_PER_CLASS   = 30;   // samples collected per activity window
const int   LOCAL_EPOCHS              = 6;
const float LEARNING_RATE             = 0.01f;
const int   ACTIVE_CLASSES            = 2;    // how many classes to train per round (1–OUTPUT_DIM)
const int   CLASS_TRANSITION_DELAY_MS = 5000; // pause between classes so user can reposition

// ── TinyML / TFLM config (inference-only) ──────────────────────────────────────
const int   TFLM_TENSOR_ARENA_SIZE   = 8 * 1024;   // 8KB arena — generous for this tiny model
const int   BENCHMARK_REPORT_EVERY_N = 20;          // upload a benchmark report every N inferences
const char* TFLITE_MODEL_URL_PATH    = "/model.tflite";

// ── Model dimensions ─────────────────────────────────────────────────────────
#define INPUT_DIM     9
#define HIDDEN1      16
#define HIDDEN2       8
#define OUTPUT_DIM    4    // one output neuron per activity class
#define TOTAL_WEIGHTS 332  // (9*16+16)+(16*8+8)+(8*4+4) = 160+136+36

// ── Activity class definitions ────────────────────────────────────────────────
// Index must match server CLASS_NAMES order
#define CLASS_STILL   0
#define CLASS_WALKING 1
#define CLASS_SHAKING 2
#define CLASS_TAPPING 3
const char* CLASS_NAMES[OUTPUT_DIM] = {"STILL", "WALKING", "SHAKING", "TAPPING"};

// One-hot label vectors for each class — used as training targets
const float ONE_HOT[OUTPUT_DIM][OUTPUT_DIM] = {
  {1, 0, 0, 0},   // STILL
  {0, 1, 0, 0},   // WALKING
  {0, 0, 1, 0},   // SHAKING
  {0, 0, 0, 1},   // TAPPING
};

// ── Network weights ───────────────────────────────────────────────────────────
float W1[INPUT_DIM][HIDDEN1], b1[HIDDEN1];
float W2[HIDDEN1][HIDDEN2],   b2[HIDDEN2];
float W3[HIDDEN2][OUTPUT_DIM],b3[OUTPUT_DIM];

// ── Client state ──────────────────────────────────────────────────────────────
enum ClientPhase { PHASE_IDLE, PHASE_TRAIN, PHASE_DOWNLOAD_MODEL, PHASE_INFERENCE };
ClientPhase clientPhase   = PHASE_IDLE;
int         currentRound  = 0;
bool        registered    = false;

// Per-epoch cross-entropy losses averaged across all classes, sent to server
float epochLosses[LOCAL_EPOCHS];

// ── TFLM state (inference-only) ────────────────────────────────────────────────
uint8_t  tfArena[TFLM_TENSOR_ARENA_SIZE];
uint8_t* tfliteModelBuffer   = nullptr;   // heap-allocated; holds raw .tflite bytes
size_t   tfliteModelSize     = 0;
bool     tfliteModelLoaded   = false;
String   serverTfliteHash    = "";        // hash announced by server in /status

tflite::MicroInterpreter* tfInterpreter = nullptr;
tflite::MicroMutableOpResolver<4>* tfResolver = nullptr;   // FC + ReLU + Softmax + Quantize/Dequantize
const tflite::Model* tfModel = nullptr;
TfLiteTensor* tfInputTensor  = nullptr;
TfLiteTensor* tfOutputTensor = nullptr;

int inferenceCount = 0;   // counts inference passes since entering PHASE_INFERENCE

// ── Xavier weight initialisation (ESP32 hardware RNG fallback) ────────────────
// Used if server is unreachable at boot. downloadModel() overwrites if successful.
void initWeightsLocal() {
  auto xav = [](int fan_in, int fan_out) -> float {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
    // esp_random() returns uint32; map to [-1000, 1000] then scale
    return ((float)((int32_t)(esp_random() % 2001) - 1000) / 1000.0f) * limit;
  };
  for (int i = 0; i < INPUT_DIM; i++)
    for (int j = 0; j < HIDDEN1; j++) W1[i][j] = xav(INPUT_DIM, HIDDEN1);
  for (int j = 0; j < HIDDEN1; j++) b1[j] = 0.0f;
  for (int i = 0; i < HIDDEN1; i++)
    for (int j = 0; j < HIDDEN2; j++) W2[i][j] = xav(HIDDEN1, HIDDEN2);
  for (int j = 0; j < HIDDEN2; j++) b2[j] = 0.0f;
  for (int i = 0; i < HIDDEN2; i++)
    for (int j = 0; j < OUTPUT_DIM; j++) W3[i][j] = xav(HIDDEN2, OUTPUT_DIM);
  for (int j = 0; j < OUTPUT_DIM; j++) b3[j] = 0.0f;
  Serial.println("[FL] Weights initialised locally via Xavier + ESP32 RNG");
}

// ── Weight serialisation ──────────────────────────────────────────────────────
void flattenWeights(float* flat) {
  int idx = 0;
  for (int i = 0; i < INPUT_DIM; i++)
    for (int j = 0; j < HIDDEN1; j++) flat[idx++] = W1[i][j];
  for (int j = 0; j < HIDDEN1; j++) flat[idx++] = b1[j];
  for (int i = 0; i < HIDDEN1; i++)
    for (int j = 0; j < HIDDEN2; j++) flat[idx++] = W2[i][j];
  for (int j = 0; j < HIDDEN2; j++) flat[idx++] = b2[j];
  for (int i = 0; i < HIDDEN2; i++)
    for (int j = 0; j < OUTPUT_DIM; j++) flat[idx++] = W3[i][j];
  for (int j = 0; j < OUTPUT_DIM; j++) flat[idx++] = b3[j];
}

void unflattenWeights(const float* flat) {
  int idx = 0;
  for (int i = 0; i < INPUT_DIM; i++)
    for (int j = 0; j < HIDDEN1; j++) W1[i][j] = flat[idx++];
  for (int j = 0; j < HIDDEN1; j++) b1[j] = flat[idx++];
  for (int i = 0; i < HIDDEN1; i++)
    for (int j = 0; j < HIDDEN2; j++) W2[i][j] = flat[idx++];
  for (int j = 0; j < HIDDEN2; j++) b2[j] = flat[idx++];
  for (int i = 0; i < HIDDEN2; i++)
    for (int j = 0; j < OUTPUT_DIM; j++) W3[i][j] = flat[idx++];
  for (int j = 0; j < OUTPUT_DIM; j++) b3[j] = flat[idx++];
}

// ── JSON helpers ──────────────────────────────────────────────────────────────
// Parse 332 weights from a nested JSON array, plus optional "round" field.
bool parseWeightsFromJson(const String& payload, float* weights_out, int& round_out) {
  int ri = payload.indexOf("\"round\":");
  if (ri >= 0) round_out = payload.substring(ri + 8).toInt();

  int start = payload.indexOf("\"weights\":");
  if (start < 0) return false;
  start = payload.indexOf('[', start);
  if (start < 0) return false;

  int idx = 0, pos = start + 1, len = payload.length();
  while (idx < TOTAL_WEIGHTS && pos < len) {
    char c = payload[pos];
    if (c == '[' || c == ']' || c == ' ' || c == '\n' || c == '\r' || c == ',') {
      pos++; continue;
    }
    int end = pos;
    while (end < len && payload[end] != ',' && payload[end] != ']'
           && payload[end] != '[' && payload[end] != ' ') end++;
    String token = payload.substring(pos, end);
    token.trim();
    if (token.length() > 0) weights_out[idx++] = token.toFloat();
    pos = end;
  }
  return (idx == TOTAL_WEIGHTS);
}

// ── Neural network — forward pass with SOFTMAX output ────────────────────────
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// h1_out[HIDDEN1], h2_out[HIDDEN2], out[OUTPUT_DIM] must be caller-allocated.
// out[] contains softmax probabilities summing to 1.0 after return.
void forward(const float* x, float* h1_out, float* h2_out, float* out) {
  // Hidden layer 1 — ReLU
  for (int j = 0; j < HIDDEN1; j++) {
    float s = b1[j];
    for (int i = 0; i < INPUT_DIM; i++) s += x[i] * W1[i][j];
    h1_out[j] = relu(s);
  }
  // Hidden layer 2 — ReLU
  for (int j = 0; j < HIDDEN2; j++) {
    float s = b2[j];
    for (int i = 0; i < HIDDEN1; i++) s += h1_out[i] * W2[i][j];
    h2_out[j] = relu(s);
  }
  // Output layer — Softmax
  // Subtract max logit for numerical stability before exp()
  float logits[OUTPUT_DIM];
  float max_logit = -1e9f;
  for (int j = 0; j < OUTPUT_DIM; j++) {
    float s = b3[j];
    for (int i = 0; i < HIDDEN2; i++) s += h2_out[i] * W3[i][j];
    logits[j] = s;
    if (s > max_logit) max_logit = s;
  }
  float sum_exp = 0.0f;
  for (int j = 0; j < OUTPUT_DIM; j++) {
    out[j] = expf(logits[j] - max_logit);
    sum_exp += out[j];
  }
  for (int j = 0; j < OUTPUT_DIM; j++) out[j] /= sum_exp;
}

// Run forward pass and return predicted class index (argmax of softmax output)
int infer(const float* x) {
  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);
  int best = 0;
  for (int j = 1; j < OUTPUT_DIM; j++)
    if (out[j] > out[best]) best = j;
  return best;
}

// ── SGD step with CROSS-ENTROPY + SOFTMAX combined gradient ──────────────────
// The combined gradient simplifies to: dL/d(logit_j) = softmax_j - y_j
// where y_j is the one-hot label. No separate softmax backward needed.
void sgd_step(const float* x, const float* one_hot_label) {
  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);

  // Output gradient: softmax_j - y_j
  float dout[OUTPUT_DIM];
  for (int j = 0; j < OUTPUT_DIM; j++)
    dout[j] = out[j] - one_hot_label[j];

  // W3, b3 gradients + propagate to h2
  float dh2[HIDDEN2] = {};
  for (int i = 0; i < HIDDEN2; i++) {
    for (int j = 0; j < OUTPUT_DIM; j++) {
      W3[i][j] -= LEARNING_RATE * dout[j] * h2[i];
      dh2[i]   += dout[j] * W3[i][j];
    }
  }
  for (int j = 0; j < OUTPUT_DIM; j++) b3[j] -= LEARNING_RATE * dout[j];

  // ReLU gate on h2
  for (int i = 0; i < HIDDEN2; i++) dh2[i] = (h2[i] > 0.0f) ? dh2[i] : 0.0f;

  // W2, b2 gradients + propagate to h1
  float dh1[HIDDEN1] = {};
  for (int i = 0; i < HIDDEN1; i++) {
    for (int j = 0; j < HIDDEN2; j++) {
      W2[i][j] -= LEARNING_RATE * dh2[j] * h1[i];
      dh1[i]   += dh2[j] * W2[i][j];
    }
  }
  for (int j = 0; j < HIDDEN2; j++) b2[j] -= LEARNING_RATE * dh2[j];

  // ReLU gate on h1
  for (int i = 0; i < HIDDEN1; i++) dh1[i] = (h1[i] > 0.0f) ? dh1[i] : 0.0f;

  // W1, b1 gradients
  for (int i = 0; i < INPUT_DIM; i++)
    for (int j = 0; j < HIDDEN1; j++)
      W1[i][j] -= LEARNING_RATE * dh1[j] * x[i];
  for (int j = 0; j < HIDDEN1; j++) b1[j] -= LEARNING_RATE * dh1[j];
}

// Cross-entropy loss for one sample: -log(softmax[true_class])
float cross_entropy(const float* x, int true_class) {
  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);
  float p = out[true_class];
  if (p < 1e-7f) p = 1e-7f;   // clamp to avoid log(0)
  return -logf(p);
}

// ── MPU6050 sensor read ───────────────────────────────────────────────────────
// Returns 9-element feature vector:
//   [0-2] accelerometer (g)      normalised by ±2g range (16384 LSB/g)
//   [3-5] gyroscope (normalised) at ±250 deg/s range (131 LSB/deg/s)
//   [6-8] reserved zeros — node-A has no additional sensors
bool read_sensors(float* out9) {
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(0x68, 14);
  if (Wire.available() < 14) return false;

  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();   // skip internal temperature bytes
  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  out9[0] = ax / 16384.0f;          // accel X  (g)
  out9[1] = ay / 16384.0f;          // accel Y  (g)
  out9[2] = az / 16384.0f;          // accel Z  (g)
  out9[3] = gx / 131.0f / 250.0f;   // gyro  X  (normalised)
  out9[4] = gy / 131.0f / 250.0f;   // gyro  Y  (normalised)
  out9[5] = gz / 131.0f / 250.0f;   // gyro  Z  (normalised)
  out9[6] = 0.0f;                    // reserved
  out9[7] = 0.0f;                    // reserved
  out9[8] = 0.0f;                    // reserved
  return true;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────
String serverUrl(const char* path) {
  return String("http://") + SERVER_IP + ":" + SERVER_PORT + path;
}

bool httpRegister() {
  HTTPClient http;
  http.begin(serverUrl("/register"));
  http.addHeader("Content-Type", "application/json");
  String body = "{\"client_id\":\"";
  body += CLIENT_ID;
  body += "\"}";
  int code = http.POST(body);
  http.end();
  if (code == 200) {
    Serial.printf("[FL] Registered as %s\n", CLIENT_ID);
    return true;
  }
  Serial.printf("[FL] Registration failed HTTP %d — retrying\n", code);
  return false;
}

// Poll /status — fills weights_out and round_out if non-null.
String pollStatus(float* weights_out = nullptr, int* round_out = nullptr) {
  HTTPClient http;
  String url = serverUrl("/status");
  url += "?client_id=";
  url += CLIENT_ID;
  http.begin(url);
  int code = http.GET();
  if (code != 200) {
    http.end();
    Serial.printf("[FL] Poll failed HTTP %d\n", code);
    return "wait";
  }
  String payload = http.getString();
  http.end();

  // Extract instruction string
  int ii = payload.indexOf("\"instruction\":\"");
  if (ii < 0) return "wait";
  int is = ii + 15;
  int ie = payload.indexOf("\"", is);
  String instruction = payload.substring(is, ie);

  // Extract round number
  if (round_out) {
    int ri = payload.indexOf("\"round\":");
    if (ri >= 0) *round_out = payload.substring(ri + 8).toInt();
  }

  // Extract tflite_hash if present (sent during download_model / inference instructions)
  int th = payload.indexOf("\"tflite_hash\":\"");
  if (th >= 0) {
    int hs = th + 16;
    int he = payload.indexOf("\"", hs);
    serverTfliteHash = payload.substring(hs, he);
  }

  // Extract weights if caller wants them
  if (weights_out && payload.indexOf("\"weights\"") >= 0) {
    int dummy = 0;
    parseWeightsFromJson(payload, weights_out, dummy);
  }

  return instruction;
}

bool downloadModel() {
  HTTPClient http;
  http.begin(serverUrl("/model"));
  int code = http.GET();
  if (code != 200) {
    http.end();
    Serial.println("[FL] Model download failed — using local weights");
    return false;
  }
  String payload = http.getString();
  http.end();

  float flat[TOTAL_WEIGHTS];
  int round = 0;
  if (parseWeightsFromJson(payload, flat, round)) {
    unflattenWeights(flat);
    currentRound = round;
    Serial.printf("[FL] Model downloaded (round %d)\n", currentRound);
    return true;
  }
  Serial.println("[FL] Model parse failed");
  return false;
}

bool uploadWeights(int n_samples) {
  float flat[TOTAL_WEIGHTS];
  flattenWeights(flat);

  String body = "{\"client_id\":\"";
  body += CLIENT_ID;
  body += "\",\"round\":";
  body += String(currentRound);
  body += ",\"n_samples\":";
  body += String(n_samples);

  // Per-epoch cross-entropy losses
  body += ",\"epoch_losses\":[";
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    body += String(epochLosses[e], 5);
    if (e < LOCAL_EPOCHS - 1) body += ",";
  }
  body += "]";

  // Flattened weights
  body += ",\"weights\":[";
  for (int i = 0; i < TOTAL_WEIGHTS; i++) {
    body += String(flat[i], 6);
    if (i < TOTAL_WEIGHTS - 1) body += ",";
  }
  body += "]}";

  HTTPClient http;
  http.begin(serverUrl("/update"));
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(15000);
  int code = http.POST(body);
  String resp = http.getString();
  http.end();

  if (code == 200) {
    Serial.println("[FL] Upload OK ✓");
    return true;
  }
  Serial.printf("[FL] Upload FAILED HTTP %d — %s\n", code, resp.c_str());
  return false;
}

// ── Weight verification ───────────────────────────────────────────────────────
void verifyWeightsWithServer() {
  Serial.println("[FL] Verifying local weights == server global weights …");
  float flat[TOTAL_WEIGHTS];
  flattenWeights(flat);

  String body = "{\"client_id\":\"";
  body += CLIENT_ID;
  body += "\",\"weights\":[";
  for (int i = 0; i < TOTAL_WEIGHTS; i++) {
    body += String(flat[i], 6);
    if (i < TOTAL_WEIGHTS - 1) body += ",";
  }
  body += "]}";

  HTTPClient http;
  http.begin(serverUrl("/verify_weights"));
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(10000);
  int code = http.POST(body);
  String resp = http.getString();
  http.end();

  if (code == 200) {
    bool matched = resp.indexOf("\"match\":true") >= 0;
    Serial.printf("[FL] Weight verification: %s\n",
                  matched ? "✓ PASSED — local == server" : "✗ FAILED — mismatch!");
  } else {
    Serial.printf("[FL] Verification request failed HTTP %d\n", code);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// TinyML (TFLM) — inference-only additions. Nothing above this point in the
// training pipeline is touched; these functions only run during the
// PHASE_DOWNLOAD_MODEL and PHASE_INFERENCE states.
// ════════════════════════════════════════════════════════════════════════════

// Compute SHA256 of a byte buffer, return as lowercase hex string
String sha256Hex(const uint8_t* data, size_t len) {
  unsigned char hash[32];
  mbedtls_sha256_context ctx;
  mbedtls_sha256_init(&ctx);
  mbedtls_sha256_starts(&ctx, 0);   // 0 = SHA-256 (not SHA-224)
  mbedtls_sha256_update(&ctx, data, len);
  mbedtls_sha256_finish(&ctx, hash);
  mbedtls_sha256_free(&ctx);

  char hexStr[65];
  for (int i = 0; i < 32; i++) sprintf(hexStr + i * 2, "%02x", hash[i]);
  hexStr[64] = '\0';
  return String(hexStr);
}

// Download the .tflite binary from the server into a heap buffer.
// Returns true on success; fills tfliteModelBuffer / tfliteModelSize.
bool downloadTfliteModel() {
  Serial.println("[TFLM] Downloading .tflite model …");
  HTTPClient http;
  http.begin(serverUrl(TFLITE_MODEL_URL_PATH));
  int code = http.GET();
  if (code != 200) {
    Serial.printf("[TFLM] Download failed HTTP %d\n", code);
    http.end();
    return false;
  }

  int len = http.getSize();
  if (len <= 0) {
    Serial.println("[TFLM] Invalid content length");
    http.end();
    return false;
  }

  // Free any previous buffer before allocating a new one
  if (tfliteModelBuffer != nullptr) {
    free(tfliteModelBuffer);
    tfliteModelBuffer = nullptr;
  }
  tfliteModelBuffer = (uint8_t*)malloc(len);
  if (tfliteModelBuffer == nullptr) {
    Serial.println("[TFLM] malloc failed — out of heap");
    http.end();
    return false;
  }

  WiFiClient* stream = http.getStreamPtr();
  int bytesRead = 0;
  unsigned long startMs = millis();
  while (bytesRead < len && (millis() - startMs) < 15000) {
    if (stream->available()) {
      int n = stream->readBytes(tfliteModelBuffer + bytesRead, len - bytesRead);
      bytesRead += n;
    }
    delay(1);
  }
  http.end();

  if (bytesRead != len) {
    Serial.printf("[TFLM] Incomplete download: %d / %d bytes\n", bytesRead, len);
    free(tfliteModelBuffer);
    tfliteModelBuffer = nullptr;
    return false;
  }

  tfliteModelSize = len;
  Serial.printf("[TFLM] Downloaded %d bytes\n", len);
  return true;
}

// Build the TFLM interpreter from the downloaded model buffer.
// Must be called once after a successful, hash-verified download.
bool setupTfliteInterpreter() {
  tfModel = tflite::GetModel(tfliteModelBuffer);
  if (tfModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[TFLM] Schema version mismatch: model=%d expected=%d\n",
                  tfModel->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Only register the ops this tiny model actually uses — keeps flash small
  static tflite::MicroMutableOpResolver<4> resolver;
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddQuantize();   // present if converter inserted quantize/dequantize ops

  static tflite::MicroInterpreter staticInterpreter(
      tfModel, resolver, tfArena, TFLM_TENSOR_ARENA_SIZE);
  tfInterpreter = &staticInterpreter;

  TfLiteStatus allocStatus = tfInterpreter->AllocateTensors();
  if (allocStatus != kTfLiteOk) {
    Serial.println("[TFLM] AllocateTensors() failed — arena too small?");
    return false;
  }

  tfInputTensor  = tfInterpreter->input(0);
  tfOutputTensor = tfInterpreter->output(0);

  Serial.printf("[TFLM] Interpreter ready — arena used: %d / %d bytes\n",
                tfInterpreter->arena_used_bytes(), TFLM_TENSOR_ARENA_SIZE);
  return true;
}

// Full pipeline: download .tflite, verify SHA256 against server hash,
// build interpreter, confirm with server via /confirm_model.
// Returns true only if EVERY step succeeds — this is the integrity gate
// that ensures this client uses the exact same model as the server.
bool acquireAndVerifyTfliteModel() {
  if (!downloadTfliteModel()) return false;

  String localHash = sha256Hex(tfliteModelBuffer, tfliteModelSize);
  Serial.printf("[TFLM] Local hash:  %s\n", localHash.c_str());
  Serial.printf("[TFLM] Server hash: %s\n", serverTfliteHash.c_str());

  bool hashMatch = localHash.equalsIgnoreCase(serverTfliteHash);
  if (!hashMatch) {
    Serial.println("[TFLM] ✗ HASH MISMATCH — refusing to load model");
  }

  // Report confirmation to server regardless of match — server tracks per-
  // client status and will not release "inference" until ALL clients match
  HTTPClient http;
  http.begin(serverUrl("/confirm_model"));
  http.addHeader("Content-Type", "application/json");
  String body = "{\"client_id\":\"";
  body += CLIENT_ID;
  body += "\",\"tflite_hash\":\"";
  body += localHash;
  body += "\"}";
  int code = http.POST(body);
  String resp = http.getString();
  http.end();

  if (code == 200) {
    Serial.printf("[TFLM] Confirmation sent — server response: %s\n", resp.c_str());
  } else {
    Serial.printf("[TFLM] Confirmation POST failed HTTP %d\n", code);
  }

  if (!hashMatch) return false;

  if (!setupTfliteInterpreter()) {
    Serial.println("[TFLM] Interpreter setup failed");
    return false;
  }

  tfliteModelLoaded = true;
  Serial.println("[TFLM] ✓ Model verified and loaded — ready for inference");
  return true;
}

// Run one TFLM forward pass. Fills outProbs[OUTPUT_DIM] (dequantized to
// float32 if the model is int8) and returns predicted class index.
// Also fills *latencyUs with wall-clock inference time in microseconds.
int tflmInfer(const float* x9, float* outProbs, unsigned long* latencyUs) {
  unsigned long t0 = micros();

  // Fill input tensor — supports both float32 and int8 quantized input
  if (tfInputTensor->type == kTfLiteFloat32) {
    for (int i = 0; i < INPUT_DIM; i++) tfInputTensor->data.f[i] = x9[i];
  } else if (tfInputTensor->type == kTfLiteInt8) {
    float scale      = tfInputTensor->params.scale;
    int   zeroPoint  = tfInputTensor->params.zero_point;
    for (int i = 0; i < INPUT_DIM; i++) {
      int32_t q = (int32_t)roundf(x9[i] / scale) + zeroPoint;
      tfInputTensor->data.int8[i] = (int8_t)constrain(q, -128, 127);
    }
  }

  TfLiteStatus invokeStatus = tfInterpreter->Invoke();
  *latencyUs = micros() - t0;

  if (invokeStatus != kTfLiteOk) {
    Serial.println("[TFLM] Invoke() failed");
    for (int j = 0; j < OUTPUT_DIM; j++) outProbs[j] = 0.0f;
    return -1;
  }

  // Read output tensor — dequantize if int8
  if (tfOutputTensor->type == kTfLiteFloat32) {
    for (int j = 0; j < OUTPUT_DIM; j++) outProbs[j] = tfOutputTensor->data.f[j];
  } else if (tfOutputTensor->type == kTfLiteInt8) {
    float scale     = tfOutputTensor->params.scale;
    int   zeroPoint = tfOutputTensor->params.zero_point;
    for (int j = 0; j < OUTPUT_DIM; j++)
      outProbs[j] = (tfOutputTensor->data.int8[j] - zeroPoint) * scale;
  }

  int best = 0;
  for (int j = 1; j < OUTPUT_DIM; j++)
    if (outProbs[j] > outProbs[best]) best = j;
  return best;
}

// Upload a benchmark comparison report to the server
void uploadBenchmarkReport(unsigned long hwLatencyUs, unsigned long tflmLatencyUs,
                            float hwConfidence, float tflmConfidence,
                            int hwClass, int tflmClass) {
  String body = "{\"client_id\":\"";
  body += CLIENT_ID;
  body += "\",\"handwritten_latency_us\":";
  body += String(hwLatencyUs);
  body += ",\"tflm_latency_us\":";
  body += String(tflmLatencyUs);
  // RAM footprint: hand-written model uses static global arrays — compute
  // analytically. TFLM arena usage is read from the interpreter.
  body += ",\"handwritten_ram_bytes\":";
  body += String((int)(TOTAL_WEIGHTS * sizeof(float)));
  body += ",\"tflm_arena_bytes\":";
  body += String((int)(tfInterpreter ? tfInterpreter->arena_used_bytes() : 0));
  body += ",\"handwritten_confidence\":";
  body += String(hwConfidence, 4);
  body += ",\"tflm_confidence\":";
  body += String(tflmConfidence, 4);
  body += ",\"predictions_match\":";
  body += (hwClass == tflmClass) ? "1" : "0";
  body += ",\"predicted_class\":";
  body += String(hwClass);
  body += "}";

  HTTPClient http;
  http.begin(serverUrl("/benchmark_report"));
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(8000);
  int code = http.POST(body);
  http.end();

  if (code == 200) {
    Serial.println("[Benchmark] Report uploaded ✓");
  } else {
    Serial.printf("[Benchmark] Upload failed HTTP %d\n", code);
  }
}


// All class samples are collected first, then shuffled together so every epoch
// sees gradients from all classes, preventing the last class from overwriting.
void runLocalTraining() {
  Serial.println("\n[FL] ── Training started (interleaved classes) ──");

  // 1. Download latest global model; fall back to existing local weights
  if (!downloadModel()) {
    Serial.println("[FL] Using existing local weights");
  }

  const int TOTAL_SAMPLES = ACTIVE_CLASSES * LOCAL_SAMPLES_PER_CLASS;
  float samples[TOTAL_SAMPLES][INPUT_DIM];
  int   labels[TOTAL_SAMPLES];

  // 2. Collect samples per-class (user performs each activity in order)
  int sampleIdx = 0;
  for (int cls = 0; cls < ACTIVE_CLASSES; cls++) {
    Serial.printf("\n[FL] ── Class %d/%d: %s ──\n",
                  cls + 1, ACTIVE_CLASSES, CLASS_NAMES[cls]);
    Serial.printf("[FL] >>> Perform %s now! Collecting %d samples …\n",
                  CLASS_NAMES[cls], LOCAL_SAMPLES_PER_CLASS);

    int collected = 0;
    while (collected < LOCAL_SAMPLES_PER_CLASS) {
      float s[INPUT_DIM];
      if (read_sensors(s)) {
        memcpy(samples[sampleIdx], s, sizeof(s));
        labels[sampleIdx] = cls;
        sampleIdx++;
        collected++;
        if (collected % 10 == 0)
          Serial.printf("  … %d / %d\n", collected, LOCAL_SAMPLES_PER_CLASS);
      }
      delay(SAMPLE_INTERVAL_MS);
    }
    Serial.printf("[FL] Collected %d samples for %s\n", collected, CLASS_NAMES[cls]);

    // Pause so user can reposition (skip after last class)
    if (cls < ACTIVE_CLASSES - 1) {
      Serial.printf("[FL] Next class in %d s — get ready …\n",
                    CLASS_TRANSITION_DELAY_MS / 1000);
      delay(CLASS_TRANSITION_DELAY_MS);
    }
  }

  // 3. Fisher-Yates shuffle — mix all class samples together
  Serial.println("\n[FL] Shuffling samples …");
  for (int i = TOTAL_SAMPLES - 1; i > 0; i--) {
    int j = esp_random() % (i + 1);
    float temp[INPUT_DIM];
    memcpy(temp, samples[i], sizeof(temp));
    memcpy(samples[i], samples[j], sizeof(samples[i]));
    memcpy(samples[j], temp, sizeof(temp));
    int t = labels[i];
    labels[i] = labels[j];
    labels[j] = t;
  }

  // 4. Train — each epoch sees all classes interleaved
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    float total_loss = 0.0f;
    for (int t = 0; t < TOTAL_SAMPLES; t++) {
      sgd_step(samples[t], ONE_HOT[labels[t]]);
      total_loss += cross_entropy(samples[t], labels[t]);
    }
    epochLosses[e] = total_loss / TOTAL_SAMPLES;
    Serial.printf("  Epoch %d/%d  CE-loss=%.5f\n", e + 1, LOCAL_EPOCHS, epochLosses[e]);
  }

  // 5. Accuracy on full training set
  int correct = 0;
  for (int t = 0; t < TOTAL_SAMPLES; t++)
    if (infer(samples[t]) == labels[t]) correct++;
  Serial.printf("[FL] Training accuracy: %d/%d (%.1f%%)\n",
                correct, TOTAL_SAMPLES, 100.0f * correct / TOTAL_SAMPLES);

  // 6. Upload final weights to server
  uploadWeights(TOTAL_SAMPLES);
  Serial.println("[FL] ── Training complete ──");
}

// ── Inference — DUAL: hand-written forward pass + TFLM, benchmarked ───────────
// Both methods run on the SAME sensor sample for a fair comparison. Every
// BENCHMARK_REPORT_EVERY_N inferences, a comparison report is uploaded.
void runInference() {
  float x[INPUT_DIM];
  if (!read_sensors(x)) return;

  // ── (a) Existing hand-written float32 inference — UNCHANGED logic ──
  unsigned long hwStart = micros();
  float h1[HIDDEN1], h2[HIDDEN2], hwProbs[OUTPUT_DIM];
  forward(x, h1, h2, hwProbs);
  unsigned long hwLatencyUs = micros() - hwStart;
  int hwPredicted = 0;
  for (int j = 1; j < OUTPUT_DIM; j++)
    if (hwProbs[j] > hwProbs[hwPredicted]) hwPredicted = j;

  Serial.printf("[INF-HW]   %-8s (%.2f%%)  %lu us  probs=[%.3f %.3f %.3f %.3f]\n",
                CLASS_NAMES[hwPredicted], 100.0f * hwProbs[hwPredicted], hwLatencyUs,
                hwProbs[0], hwProbs[1], hwProbs[2], hwProbs[3]);

  // ── (b) TFLM inference — only if model loaded and verified ──
  if (tfliteModelLoaded) {
    float tflmProbs[OUTPUT_DIM];
    unsigned long tflmLatencyUs = 0;
    int tflmPredicted = tflmInfer(x, tflmProbs, &tflmLatencyUs);

    if (tflmPredicted >= 0) {
      Serial.printf("[INF-TFLM] %-8s (%.2f%%)  %lu us  probs=[%.3f %.3f %.3f %.3f]\n",
                    CLASS_NAMES[tflmPredicted], 100.0f * tflmProbs[tflmPredicted], tflmLatencyUs,
                    tflmProbs[0], tflmProbs[1], tflmProbs[2], tflmProbs[3]);

      const char* agree = (hwPredicted == tflmPredicted) ? "✓ AGREE" : "✗ DISAGREE";
      float speedup = (tflmLatencyUs > 0) ? (float)hwLatencyUs / (float)tflmLatencyUs : 0.0f;
      Serial.printf("[INF-CMP]  %s  |  speedup=%.2fx  |  HW_RAM=%dB  TFLM_arena=%dB\n",
                    agree, speedup,
                    (int)(TOTAL_WEIGHTS * sizeof(float)),
                    (int)tfInterpreter->arena_used_bytes());

      inferenceCount++;
      if (inferenceCount % BENCHMARK_REPORT_EVERY_N == 0) {
        uploadBenchmarkReport(hwLatencyUs, tflmLatencyUs,
                              hwProbs[hwPredicted], tflmProbs[tflmPredicted],
                              hwPredicted, tflmPredicted);
      }
    }
  } else {
    Serial.println("[INF-TFLM] (not loaded — running hand-written only)");
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[Boot] FL Activity Classifier — ESP32");

  // Initialise MPU6050
  Wire.begin(21, 22);
  Wire.beginTransmission(0x68);
  Wire.write(0x6B); Wire.write(0x00);   // wake from sleep
  Wire.endTransmission();
  Serial.println("[MPU] MPU6050 awake");

  // Local Xavier weight fallback — overwritten by server model if reachable
  initWeightsLocal();

  // Connect WiFi
  Serial.print("[WiFi] Connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());

  // Register with FL server — retry indefinitely
  while (!registered) {
    registered = httpRegister();
    if (!registered) delay(3000);
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  // WiFi watchdog
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Reconnecting …");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  switch (clientPhase) {

    // ── IDLE: poll server, wait for instruction ──────────────────────────────
    case PHASE_IDLE: {
      int serverRound = 0;
      float weights[TOTAL_WEIGHTS];
      String instr = pollStatus(weights, &serverRound);

      Serial.printf("[FL] IDLE — instruction=%s  round=%d\n",
                    instr.c_str(), serverRound);

      if (instr == "train") {
        clientPhase = PHASE_TRAIN;

      } else if (instr == "download_model") {
        // FL converged; TFLite ready on server but THIS client hasn't
        // confirmed the hash yet — load hand-written weights as fallback,
        // then move to the dedicated download/verify phase
        unflattenWeights(weights);
        currentRound = serverRound;
        clientPhase = PHASE_DOWNLOAD_MODEL;

      } else if (instr == "inference") {
        // All clients (including this one) already confirmed the model
        // hash in a previous cycle — just load weights and go straight to
        // dual inference
        Serial.println("[FL] → INFERENCE MODE (model already verified)");
        unflattenWeights(weights);
        currentRound = serverRound;
        if (!tfliteModelLoaded) {
          // Edge case: client restarted after acking — reacquire model
          acquireAndVerifyTfliteModel();
        }
        inferenceCount = 0;
        clientPhase = PHASE_INFERENCE;

      } else {
        // "wait" — poll again after interval
        delay(POLL_INTERVAL_MS);
      }
      break;
    }

    // ── TRAIN: collect labelled samples, train, upload ───────────────────────
    case PHASE_TRAIN: {
      runLocalTraining();
      clientPhase = PHASE_IDLE;
      delay(1000);   // brief pause before next poll
      break;
    }

    // ── DOWNLOAD_MODEL: fetch .tflite, verify hash, confirm with server ──────
    case PHASE_DOWNLOAD_MODEL: {
      Serial.println("\n[FL] ── FL converged — acquiring TFLite model ──");
      bool ok = acquireAndVerifyTfliteModel();
      if (ok) {
        Serial.println("[FL] Model verified — waiting for other clients …");
      } else {
        Serial.println("[FL] Model verification failed — will retry on next poll");
      }
      // Either way, return to idle; server will re-send "download_model"
      // until this client's ack is recorded, then send "inference" once
      // ALL clients have acked
      clientPhase = PHASE_IDLE;
      delay(POLL_INTERVAL_MS);
      break;
    }

    // ── INFERENCE: run dual forward pass (hand-written + TFLM), benchmark ────
    case PHASE_INFERENCE: {
      runInference();
      delay(INFERENCE_INTERVAL_MS);

      // Check every 60 s if a new FL cycle has started
      static unsigned long lastCycleCheck = 0;
      if (millis() - lastCycleCheck > 60000UL) {
        lastCycleCheck = millis();
        String instr = pollStatus();
        if (instr == "train") {
          Serial.println("[FL] New FL cycle detected — returning to IDLE");
          tfliteModelLoaded = false;   // model will be replaced; force re-verify next cycle
          clientPhase = PHASE_IDLE;
        }
      }
      break;
    }
  }
}
