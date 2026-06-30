/*
 * Model Configuration — Shared constants and weight declarations
 *
 * Change this file when modifying model architecture.
 * Both training.h and inference.h depend on these definitions.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <Arduino.h>
#include "esp_random.h"

// ── Model dimensions ─────────────────────────────────────────────────────────
#define INPUT_DIM     9
#define HIDDEN1      16
#define HIDDEN2       8
#define OUTPUT_DIM    4    // one output neuron per activity class
#define TOTAL_WEIGHTS 332  // (9*16+16)+(16*8+8)+(8*4+4) = 160+136+36

// ── Activity class definitions ────────────────────────────────────────────────
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

// ── Network weights (global) ─────────────────────────────────────────────────
float W1[INPUT_DIM][HIDDEN1], b1[HIDDEN1];
float W2[HIDDEN1][HIDDEN2],   b2[HIDDEN2];
float W3[HIDDEN2][OUTPUT_DIM],b3[OUTPUT_DIM];

// ── Xavier weight initialisation ─────────────────────────────────────────────
inline void initWeightsLocal() {
  auto xav = [](int fan_in, int fan_out) -> float {
    float limit = sqrtf(6.0f / (fan_in + fan_out));
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

// ── Weight serialisation ─────────────────────────────────────────────────────
inline void flattenWeights(float* flat) {
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

inline void unflattenWeights(const float* flat) {
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

#endif
