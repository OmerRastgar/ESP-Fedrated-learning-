/*
 * Training Module — SGD, loss functions, and training loop
 *
 * Swap this file to change training approach (e.g., ESN, different optimizer).
 * Must implement:
 *   - sgd_step(x, one_hot_label) — single weight update
 *   - cross_entropy(x, true_class) — compute loss
 *   - runLocalTraining() — full training loop with data collection
 *
 * Dependencies: inference.h (for forward pass), model_config.h
 */

#ifndef TRAINING_H
#define TRAINING_H

#include "model_config.h"
#include "inference.h"

// ── Training hyperparameters ─────────────────────────────────────────────────
// These can be moved to main config if needed
#ifndef LOCAL_EPOCHS
#define LOCAL_EPOCHS 6
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01f
#endif

#ifndef ACTIVE_CLASSES
#define ACTIVE_CLASSES 2
#endif

#ifndef LOCAL_SAMPLES_PER_CLASS
#define LOCAL_SAMPLES_PER_CLASS 30
#endif

#ifndef SAMPLE_INTERVAL_MS
#define SAMPLE_INTERVAL_MS 200
#endif

#ifndef CLASS_TRANSITION_DELAY_MS
#define CLASS_TRANSITION_DELAY_MS 5000
#endif

// ── Per-epoch losses (shared with main for upload) ───────────────────────────
float epochLosses[LOCAL_EPOCHS];

// ── SGD step with CROSS-ENTROPY + SOFTMAX combined gradient ──────────────────
// Gradient simplifies to: dL/d(logit_j) = softmax_j - y_j
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

// ── Cross-entropy loss ───────────────────────────────────────────────────────
float cross_entropy(const float* x, int true_class) {
  float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
  forward(x, h1, h2, out);
  float p = out[true_class];
  if (p < 1e-7f) p = 1e-7f;
  return -logf(p);
}

// ── Data collection + training loop ──────────────────────────────────────────
// Requires read_sensors() from main file or sensor module
extern bool read_sensors(float* out9);
extern bool downloadModel();
extern bool uploadWeights(int n_samples);

void runLocalTraining() {
  Serial.println("\n[FL] Training started (interleaved classes)");

  // 1. Download latest global model
  if (!downloadModel()) {
    Serial.println("[FL] Using existing local weights");
  }

  const int TOTAL_SAMPLES = ACTIVE_CLASSES * LOCAL_SAMPLES_PER_CLASS;
  float samples[TOTAL_SAMPLES][INPUT_DIM];
  int   labels[TOTAL_SAMPLES];

  // 2. Collect samples per-class
  int sampleIdx = 0;
  for (int cls = 0; cls < ACTIVE_CLASSES; cls++) {
    Serial.printf("\n[FL] Class %d/%d: %s\n", cls + 1, ACTIVE_CLASSES, CLASS_NAMES[cls]);
    Serial.printf("[FL] Perform %s now! Collecting %d samples...\n",
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
          Serial.printf("  %d / %d\n", collected, LOCAL_SAMPLES_PER_CLASS);
      }
      delay(SAMPLE_INTERVAL_MS);
    }
    Serial.printf("[FL] Collected %d samples for %s\n", collected, CLASS_NAMES[cls]);

    if (cls < ACTIVE_CLASSES - 1) {
      Serial.printf("[FL] Next class in %d s...\n", CLASS_TRANSITION_DELAY_MS / 1000);
      delay(CLASS_TRANSITION_DELAY_MS);
    }
  }

  // 3. Shuffle samples
  Serial.println("\n[FL] Shuffling samples...");
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

  // 4. Train epochs
  for (int e = 0; e < LOCAL_EPOCHS; e++) {
    float total_loss = 0.0f;
    for (int t = 0; t < TOTAL_SAMPLES; t++) {
      sgd_step(samples[t], ONE_HOT[labels[t]]);
      total_loss += cross_entropy(samples[t], labels[t]);
    }
    epochLosses[e] = total_loss / TOTAL_SAMPLES;
    Serial.printf("  Epoch %d/%d  CE-loss=%.5f\n", e + 1, LOCAL_EPOCHS, epochLosses[e]);
  }

  // 5. Accuracy check
  int correct = 0;
  for (int t = 0; t < TOTAL_SAMPLES; t++)
    if (infer(samples[t]) == labels[t]) correct++;
  Serial.printf("[FL] Training accuracy: %d/%d (%.1f%%)\n",
                correct, TOTAL_SAMPLES, 100.0f * correct / TOTAL_SAMPLES);

  // 6. Upload weights
  uploadWeights(TOTAL_SAMPLES);
  Serial.println("[FL] Training complete");
}

#endif
