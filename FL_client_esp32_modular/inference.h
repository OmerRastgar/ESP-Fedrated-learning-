/*
 * Inference Module — Forward pass and prediction
 *
 * Swap this file to change inference approach (e.g., TFLite Micro, different activation).
 * Must implement:
 *   - forward(x, h1_out, h2_out, out) — full forward pass
 *   - infer(x) — returns predicted class index
 *   - runInference() — reads sensor, prints prediction
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_config.h"

// ── Activation functions ─────────────────────────────────────────────────────
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// ── Forward pass with SOFTMAX output ─────────────────────────────────────────
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

// ── Predict class index (argmax) ─────────────────────────────────────────────
int infer(const float* x) {
  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);
  int best = 0;
  for (int j = 1; j < OUTPUT_DIM; j++)
    if (out[j] > out[best]) best = j;
  return best;
}

// ── Run inference on live sensor data ────────────────────────────────────────
// Requires read_sensors() from main file or sensor module
extern bool read_sensors(float* out9);

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

#endif
