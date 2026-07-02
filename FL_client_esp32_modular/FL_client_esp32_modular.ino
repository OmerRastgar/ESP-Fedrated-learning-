/*
 * Federated Learning Client — ESP32 (Modular Version)
 *
 * This is the main file that handles:
 *   - WiFi connection and reconnection
 *   - Server communication (register, poll, download/upload weights)
 *   - Sensor reading (MPU6050)
 *   - State machine (idle → train → inference → idle)
 *
 * Modular files:
 *   - model_config.h — Model dimensions, weights, init, serialization
 *   - inference.h    — Forward pass, prediction
 *   - training.h     — SGD, loss, data collection, training loop
 *
 * To swap model/approach: replace inference.h and/or training.h
 * Change WIFI_SSID, WIFI_PASS, SERVER_IP, CLIENT_ID before flashing.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

// ── Include modular components ───────────────────────────────────────────────
#include "model_config.h"

// Choose inference method: uncomment ONE of the following
//#include "inference.h"         // Manual forward pass (default)
#include "inference_tflite.h"  // TFLite (requires compatible TensorFlowLite library)

#include "training.h"

// ── Config ────────────────────────────────────────────────────────────────────
const char* WIFI_SSID   = "Cybergaar";
const char* WIFI_PASS   = "curedata4low";
const char* SERVER_IP   = "10.219.84.92";
const int   SERVER_PORT = 5000;
const char* CLIENT_ID   = "esp32-node-B";   // unique per device

const int   POLL_INTERVAL_MS      = 3000;
const int   INFERENCE_INTERVAL_MS = 500;

// ── Client state ──────────────────────────────────────────────────────────────
enum ClientPhase { PHASE_IDLE, PHASE_TRAIN, PHASE_INFERENCE };
ClientPhase clientPhase  = PHASE_IDLE;
int         currentRound = 0;
bool        registered   = false;

// ── MPU6050 sensor read ───────────────────────────────────────────────────────
bool read_sensors(float* out9) {
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom(0x68, 14);
  if (Wire.available() < 14) return false;

  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();   // skip temperature
  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  out9[0] = ax / 16384.0f;
  out9[1] = ay / 16384.0f;
  out9[2] = az / 16384.0f;
  out9[3] = gx / 131.0f / 250.0f;
  out9[4] = gy / 131.0f / 250.0f;
  out9[5] = gz / 131.0f / 250.0f;
  out9[6] = 0.0f;
  out9[7] = 0.0f;
  out9[8] = 0.0f;
  return true;
}

// ── JSON helper ───────────────────────────────────────────────────────────────
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
  Serial.printf("[FL] Registration failed HTTP %d\n", code);
  return false;
}

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

  int ii = payload.indexOf("\"instruction\":\"");
  if (ii < 0) return "wait";
  int is = ii + 15;
  int ie = payload.indexOf("\"", is);
  String instruction = payload.substring(is, ie);

  if (round_out) {
    int ri = payload.indexOf("\"round\":");
    if (ri >= 0) *round_out = payload.substring(ri + 8).toInt();
  }

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

  body += ",\"epoch_losses\":[";
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    body += String(epochLosses[e], 5);
    if (e < LOCAL_EPOCHS - 1) body += ",";
  }
  body += "]";

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
    Serial.println("[FL] Upload OK");
    return true;
  }
  Serial.printf("[FL] Upload FAILED HTTP %d — %s\n", code, resp.c_str());
  return false;
}

void verifyWeightsWithServer() {
  Serial.println("[FL] Verifying weights...");
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
    Serial.printf("[FL] Verification: %s\n",
                  matched ? "PASSED" : "FAILED");
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[Boot] FL Client — ESP32 (Modular)");

  // MPU6050 init (ESP32 I2C: SDA=21, SCL=22)
  Wire.begin(21, 22);
  Wire.beginTransmission(0x68);
  Wire.write(0x6B); Wire.write(0x00);
  Wire.endTransmission();
  Serial.println("[MPU] MPU6050 awake");

  initWeightsLocal();

  Serial.print("[WiFi] Connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());

  while (!registered) {
    registered = httpRegister();
    if (!registered) delay(3000);
  }
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Reconnecting...");
    WiFi.reconnect();
    delay(3000);
    return;
  }

  switch (clientPhase) {

    case PHASE_IDLE: {
      int serverRound = 0;
      float weights[TOTAL_WEIGHTS];
      String instr = pollStatus(weights, &serverRound);

      Serial.printf("[FL] IDLE — %s  round=%d\n", instr.c_str(), serverRound);

      if (instr == "train") {
        clientPhase = PHASE_TRAIN;
      } else if (instr == "inference") {
        Serial.println("[FL] INFERENCE MODE");
        unflattenWeights(weights);
        currentRound = serverRound;
        verifyWeightsWithServer();
        clientPhase = PHASE_INFERENCE;
      } else {
        delay(POLL_INTERVAL_MS);
      }
      break;
    }

    case PHASE_TRAIN: {
      runLocalTraining();  // from training.h
      // After training, weights are uploaded to server
      // Server will update TFLite model with new weights
      // TFLite model will be downloaded automatically on first inference call
      clientPhase = PHASE_IDLE;
      delay(1000);
      break;
    }

    case PHASE_INFERENCE: {
      runInference();  // from inference.h
      delay(INFERENCE_INTERVAL_MS);

      static unsigned long lastCycleCheck = 0;
      if (millis() - lastCycleCheck > 60000UL) {
        lastCycleCheck = millis();
        String instr = pollStatus();
        if (instr == "train") {
          Serial.println("[FL] New cycle — returning to IDLE");
          clientPhase = PHASE_IDLE;
        }
      }
      break;
    }
  }
}
