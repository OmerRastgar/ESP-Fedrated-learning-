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
 * Server-managed round lifecycle (clients poll /status):
 *   wait      → stay idle, poll again after POLL_INTERVAL_MS
 *   train     → download model, train all 4 classes, upload weights
 *   inference → load final weights, verify hash, run forward pass locally
 *
 * Change WIFI_SSID, WIFI_PASS, SERVER_IP, CLIENT_ID before flashing each node.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <math.h>
#include "esp_random.h"   // ESP32 hardware RNG for Xavier fallback init

// ── Config ────────────────────────────────────────────────────────────────────
const char* WIFI_SSID   = "Cybergaar";
const char* WIFI_PASS   = "curedata4low";
const char* SERVER_IP   = "10.219.84.92";
const int   SERVER_PORT = 5000;
const char* CLIENT_ID             = "esp32-node-A";   // unique per device

const int   POLL_INTERVAL_MS          = 3000;  // idle poll cadence
const int   INFERENCE_INTERVAL_MS     = 500;   // forward pass cadence in inference mode
const int   SAMPLE_INTERVAL_MS        = 200;    // ms between MPU reads during collection
const int   LOCAL_SAMPLES_PER_CLASS   = 25;   // samples collected per activity window
const int   LOCAL_EPOCHS              = 6;
const float LEARNING_RATE             = 0.01f;
const int   ACTIVE_CLASSES            = 2;    // how many classes to train per round (1–OUTPUT_DIM)
const int   CLASS_TRANSITION_DELAY_MS = 5000; // pause between classes so user can reposition

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
enum ClientPhase { PHASE_IDLE, PHASE_TRAIN, PHASE_INFERENCE };
ClientPhase clientPhase   = PHASE_IDLE;
int         currentRound  = 0;
bool        registered    = false;

// Per-epoch cross-entropy losses averaged across all classes, sent to server
float epochLosses[LOCAL_EPOCHS];

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

// ── Local training — interleaved classes, no catastrophic forgetting ──────────
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

// ── Inference — single forward pass, print class ──────────────────────────────
void runInference() {
  float x[INPUT_DIM];
  if (!read_sensors(x)) return;

  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);
  int predicted = 0;
  for (int j = 1; j < OUTPUT_DIM; j++)
    if (out[j] > out[predicted]) predicted = j;

  Serial.printf("[INF] %s  (%.2f%%)  probs=[%.3f %.3f %.3f %.3f]\n",
                CLASS_NAMES[predicted],
                100.0f * out[predicted],
                out[0], out[1], out[2], out[3]);
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

      } else if (instr == "inference") {
        Serial.println("[FL] → INFERENCE MODE");
        unflattenWeights(weights);   // load final converged weights from server
        currentRound = serverRound;
        verifyWeightsWithServer();   // integrity check: local must == server
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

    // ── INFERENCE: run forward pass on live sensor data ──────────────────────
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
          clientPhase = PHASE_IDLE;
        }
      }
      break;
    }
  }
}
