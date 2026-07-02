/*
 * Inference Module — Adafruit TFLite Micro version
 *
 * Uses Adafruit TensorFlow Lite library for inference.
 * Downloads updated TFLite model from server after each training round.
 *
 * Requirements:
 *   1. Install "Adafruit TensorFlow Lite" library in Arduino IDE
 *      - Arduino IDE: Sketch → Include Library → Manage Libraries
 *      - Search "Adafruit TensorFlow Lite"
 *   2. Server must have /download_tflite endpoint enabled
 *
 * To use: In FL_client_esp32_modular.ino, change include to:
 *   // #include "inference.h"
 *   #include "inference_tflite.h"
 *
 * Note: Training still uses manual SGD (training.h).
 *       After training, TFLite model is downloaded from server for inference.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_config.h"
#include <Adafruit_TFLite.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ── TFLite arena size ────────────────────────────────────────────────────────
#define TFLITE_ARENA_SIZE 8 * 1024

// ── TFLite objects ───────────────────────────────────────────────────────────
Adafruit_TFLite* tflite = nullptr;

bool tflite_initialized = false;
bool tflite_model_downloaded = false;

// ── Model buffer for downloaded TFLite model ─────────────────────────────────
#define MAX_MODEL_SIZE 10240
uint8_t tflite_model_buffer[MAX_MODEL_SIZE];
size_t tflite_model_size = 0;

// ── Server URL helper ────────────────────────────────────────────────────────
extern String serverUrl(const char* path);

// ── Download TFLite model from server ────────────────────────────────────────
bool downloadTFLiteModel() {
    Serial.println("[TFLite] Downloading model from server...");
    
    HTTPClient http;
    http.begin(serverUrl("/download_tflite"));
    http.setTimeout(10000);
    
    int code = http.GET();
    if (code != 200) {
        Serial.printf("[TFLite] Download failed: HTTP %d\n", code);
        http.end();
        return false;
    }
    
    int len = http.getSize();
    if (len <= 0 || len > MAX_MODEL_SIZE) {
        Serial.printf("[TFLite] Invalid model size: %d\n", len);
        http.end();
        return false;
    }
    
    WiFiClient* stream = http.getStreamPtr();
    size_t bytesRead = 0;
    unsigned long timeout = millis() + 10000;
    
    while (bytesRead < len && millis() < timeout) {
        size_t available = stream->available();
        if (available) {
            size_t toRead = min(available, (size_t)(len - bytesRead));
            size_t read = stream->readBytes(tflite_model_buffer + bytesRead, toRead);
            bytesRead += read;
        }
        delay(1);
    }
    
    http.end();
    
    if (bytesRead != len) {
        Serial.printf("[TFLite] Download incomplete: %d/%d bytes\n", bytesRead, len);
        return false;
    }
    
    tflite_model_size = bytesRead;
    tflite_model_downloaded = true;
    Serial.printf("[TFLite] Model downloaded: %d bytes\n", tflite_model_size);
    
    // Reset initialization flag to reload model
    tflite_initialized = false;
    
    return true;
}

// ── Initialize TFLite ────────────────────────────────────────────────────────
bool initTFLite() {
    if (!tflite_model_downloaded || tflite_model_size == 0) {
        Serial.println("[TFLite] No model downloaded");
        return false;
    }
    
    // Create TFLite object if not exists
    if (tflite == nullptr) {
        tflite = new Adafruit_TFLite(TFLITE_ARENA_SIZE);
    }
    
    // Initialize
    if (!tflite->begin()) {
        Serial.println("[TFLite] begin() failed!");
        return false;
    }
    
    // Load model from buffer
    if (!tflite->loadModel(tflite_model_buffer)) {
        Serial.println("[TFLite] loadModel() failed!");
        return false;
    }
    
    Serial.println("[TFLite] Initialized with downloaded model");
    tflite_initialized = true;
    return true;
}

// ── Forward pass using TFLite ────────────────────────────────────────────────
void forward(const float* x, float* h1_out, float* h2_out, float* out) {
    if (!tflite_initialized) {
        memset(out, 0, OUTPUT_DIM * sizeof(float));
        return;
    }
    
    // Copy input
    memcpy(tflite->input->data.f, x, INPUT_DIM * sizeof(float));
    
    // Run inference
    if (!tflite->interpreter->Invoke()) {
        Serial.println("[TFLite] Invoke() failed!");
        return;
    }
    
    // Copy output
    memcpy(out, tflite->output->data.f, OUTPUT_DIM * sizeof(float));
    
    // h1/h2 not accessible
    memset(h1_out, 0, HIDDEN1 * sizeof(float));
    memset(h2_out, 0, HIDDEN2 * sizeof(float));
}

// ── Predict class index (argmax) ─────────────────────────────────────────────
int infer(const float* x) {
    if (!tflite_initialized) return 0;
    
    memcpy(tflite->input->data.f, x, INPUT_DIM * sizeof(float));
    
    if (!tflite->interpreter->Invoke()) return 0;
    
    const float* probs = tflite->output->data.f;
    int best = 0;
    for (int j = 1; j < OUTPUT_DIM; j++)
        if (probs[j] > probs[best]) best = j;
    return best;
}

// ── Run inference on live sensor data ────────────────────────────────────────
extern bool read_sensors(float* out9);

void runInference() {
    // Download model if not done yet
    if (!tflite_model_downloaded) {
        if (!downloadTFLiteModel()) {
            Serial.println("[TFLite] Model download failed, using manual inference");
            // Fall back to manual inference
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
            return;
        }
    }
    
    // Initialize if needed
    if (!tflite_initialized) {
        if (!initTFLite()) {
            Serial.println("[TFLite] Init failed!");
            return;
        }
    }
    
    float x[INPUT_DIM];
    if (!read_sensors(x)) return;
    
    memcpy(tflite->input->data.f, x, INPUT_DIM * sizeof(float));
    
    if (!tflite->interpreter->Invoke()) {
        Serial.println("[TFLite] Invoke() failed!");
        return;
    }
    
    const float* out = tflite->output->data.f;
    
    int predicted = 0;
    for (int j = 1; j < OUTPUT_DIM; j++)
        if (out[j] > out[predicted]) predicted = j;
    
    Serial.printf("[INF-TFLite] %s  (%.2f%%)  probs=[%.3f %.3f %.3f %.3f]\n",
                  CLASS_NAMES[predicted],
                  100.0f * out[predicted],
                  out[0], out[1], out[2], out[3]);
}

#endif
