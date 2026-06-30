"""
Federated Learning Orchestration Server
- Manages round lifecycle: init → sequential client training → FedAvg → convergence check
- Clients poll /status to receive instructions
- Sequential training: one client trains at a time
- Convergence: loss below threshold OR plateau detection
- After convergence: convert final weights to TFLite, host as binary, verify
  hash with all clients, THEN release "inference" instruction.
- Inference phase: clients run BOTH hand-written and TFLM inference per cycle
  and upload a comparison report (latency/memory/accuracy) for benchmarking.
"""

import json
import time
import hashlib
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from collections import defaultdict

app = Flask(__name__)

# ─── Model Architecture ────────────────────────────────────────────────────────
INPUT_DIM   = 9
HIDDEN1     = 16
HIDDEN2     = 8
OUTPUT_DIM  = 4
TOTAL_WEIGHTS = (INPUT_DIM*HIDDEN1 + HIDDEN1) + (HIDDEN1*HIDDEN2 + HIDDEN2) + (HIDDEN2*OUTPUT_DIM + OUTPUT_DIM)

# ─── FL Hyperparameters ────────────────────────────────────────────────────────
MIN_CLIENTS          = 2          # configurable: minimum clients before a round starts
CONVERGENCE_LOSS     = 0.01       # absolute loss threshold
CONVERGENCE_PLATEAU  = 5          # rounds with < this % improvement → converged
PLATEAU_TOLERANCE    = 0.005      # minimum relative improvement to not count as plateau
MAX_ROUNDS           = 10     # hard cap
INFERENCE_COOLDOWN_S = 900        # 15 minutes between FL cycles

# ─── TFLite conversion settings ────────────────────────────────────────────────
TFLITE_MODEL_PATH    = "model.tflite"   # written to disk, served at /model.tflite
TFLITE_QUANTIZE_INT8 = True             # int8 quantisation (smaller, faster on ESP32)

# ─── State ─────────────────────────────────────────────────────────────────────
lock = threading.Lock()

def init_weights():
    rng = np.random.default_rng(42)
    shapes = [
        (INPUT_DIM, HIDDEN1), (HIDDEN1,),
        (HIDDEN1, HIDDEN2),   (HIDDEN2,),
        (HIDDEN2, OUTPUT_DIM),(OUTPUT_DIM,),
    ]
    weights = []
    for s in shapes:
        if len(s) == 2:
            fan_in, fan_out = s
            limit = np.sqrt(6 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, s).tolist())
        else:
            weights.append(np.zeros(s).tolist())
    return weights

class FLState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.global_weights   = init_weights()
        self.round_number     = 0
        self.phase            = "waiting_for_clients"  # see phases below
        # Phases:
        #   waiting_for_clients  – not enough clients registered
        #   training             – a client is actively training
        #   aggregating          – server is running FedAvg
        #   converged            – FL done, all clients in inference mode
        #   cooldown             – waiting INFERENCE_COOLDOWN_S before next cycle

        self.registered_clients  = {}        # client_id → last_seen timestamp
        self.training_order      = []        # ordered list of client_ids for this round
        self.current_trainer_idx = 0         # index into training_order
        self.round_updates       = []        # list of (weights, n_samples) collected this round
        self.round_losses        = {}        # client_id → list of per-epoch losses this round

        self.history_loss        = []        # avg loss per round (for convergence & plot)
        self.plateau_count       = 0
        self.cooldown_start      = None
        self.cycle_number        = 0

        # ── TFLite conversion state ──
        self.tflite_ready        = False      # True once .tflite written + hash computed
        self.tflite_hash         = None       # SHA256 of the .tflite file
        self.tflite_size_bytes   = 0
        self.client_model_acks   = {}         # client_id → bool (confirmed same hash)
        self.benchmark_reports   = {}         # client_id → latest comparison report

state = FLState()

# ─── Helpers ───────────────────────────────────────────────────────────────────

def federated_average(updates):
    total_samples = sum(n for _, n in updates)
    new_weights = []
    for layer_idx in range(len(updates[0][0])):
        layer_avg = np.zeros_like(np.array(updates[0][0][layer_idx], dtype=float))
        for w, n in updates:
            layer_avg += np.array(w[layer_idx], dtype=float) * (n / total_samples)
        new_weights.append(layer_avg.tolist())
    return new_weights

def weights_are_equal(w1, w2, tol=1e-5):
    """Check that two weight lists are element-wise equal within tolerance."""
    for l1, l2 in zip(w1, w2):
        a1 = np.array(l1, dtype=float)
        a2 = np.array(l2, dtype=float)
        if np.max(np.abs(a1 - a2)) > tol:
            return False
    return True

def convert_to_tflite(weights):
    """
    Reconstruct the 9→16→8→4 Keras model from FL weights and convert to
    .tflite. Uses int8 quantisation if TFLITE_QUANTIZE_INT8 is set, falling
    back to float32 if a representative dataset isn't available/needed.

    weights layout (matches client flatten/unflatten order):
      [0] W1 (9,16)  [1] b1 (16,)
      [2] W2 (16,8)  [3] b2 (8,)
      [4] W3 (8,4)   [5] b3 (4,)

    Returns (tflite_bytes, sha256_hex, size_bytes) or (None, None, 0) on failure.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[TFLite] ERROR: tensorflow not installed. Run: pip install tensorflow")
        return None, None, 0

    W1 = np.array(weights[0], dtype=np.float32)
    b1 = np.array(weights[1], dtype=np.float32)
    W2 = np.array(weights[2], dtype=np.float32)
    b2 = np.array(weights[3], dtype=np.float32)
    W3 = np.array(weights[4], dtype=np.float32)
    b3 = np.array(weights[5], dtype=np.float32)

    # Rebuild identical architecture: Dense(16,relu) → Dense(8,relu) → Dense(4,softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(HIDDEN1, activation="relu", name="dense1"),
        tf.keras.layers.Dense(HIDDEN2, activation="relu", name="dense2"),
        tf.keras.layers.Dense(OUTPUT_DIM, activation="softmax", name="dense3"),
    ])
    model.build(input_shape=(None, INPUT_DIM))
    model.get_layer("dense1").set_weights([W1, b1])
    model.get_layer("dense2").set_weights([W2, b2])
    model.get_layer("dense3").set_weights([W3, b3])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if TFLITE_QUANTIZE_INT8:
        # Representative dataset for activation range calibration — sample
        # the input domain uniformly since FL clients use normalised
        # accel/gyro features roughly in [-1, 1]
        def representative_dataset():
            rng = np.random.default_rng(0)
            for _ in range(100):
                sample = rng.uniform(-1.0, 1.0, size=(1, INPUT_DIM)).astype(np.float32)
                yield [sample]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        # Keep float32 in/out for simplicity on the ESP32 side; only internal
        # weights/activations are quantised to int8
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

    try:
        tflite_bytes = converter.convert()
    except Exception as e:
        print(f"[TFLite] Conversion failed: {e}")
        return None, None, 0

    sha256_hex = hashlib.sha256(tflite_bytes).hexdigest()
    return tflite_bytes, sha256_hex, len(tflite_bytes)

def build_and_save_tflite_model():
    """Convert state.global_weights to .tflite, write to disk, update state."""
    print("[TFLite] Converting final FL weights to .tflite …")
    tflite_bytes, sha256_hex, size_bytes = convert_to_tflite(state.global_weights)

    if tflite_bytes is None:
        print("[TFLite] Conversion FAILED — inference will fall back to hand-written model only")
        state.tflite_ready = False
        return False

    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_bytes)

    state.tflite_hash       = sha256_hex
    state.tflite_size_bytes = size_bytes
    state.tflite_ready      = True
    state.client_model_acks = {}   # reset acks — clients must re-confirm new model

    print(f"[TFLite] Saved {TFLITE_MODEL_PATH}  ({size_bytes} bytes)")
    print(f"[TFLite] SHA256: {sha256_hex}")
    return True

def check_convergence():
    """Returns (converged: bool, reason: str)"""
    if len(state.history_loss) == 0:
        return False, ""

    latest = state.history_loss[-1]

    # Absolute threshold
    if latest < CONVERGENCE_LOSS:
        return True, f"loss {latest:.5f} < threshold {CONVERGENCE_LOSS}"

    # Plateau detection
    if len(state.history_loss) >= 2:
        prev = state.history_loss[-2]
        if prev > 0:
            improvement = (prev - latest) / prev
            if improvement < PLATEAU_TOLERANCE:
                state.plateau_count += 1
            else:
                state.plateau_count = 0

        if state.plateau_count >= CONVERGENCE_PLATEAU:
            return True, f"loss plateaued for {CONVERGENCE_PLATEAU} consecutive rounds"

    return False, ""

def save_plots():
    """Save per-round loss plot."""
    if not state.history_loss:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(state.history_loss) + 1), state.history_loss,
            marker='o', linewidth=2, color='#2563eb')
    ax.axhline(CONVERGENCE_LOSS, color='#dc2626', linestyle='--', label=f'Threshold ({CONVERGENCE_LOSS})')
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Average Loss")
    ax.set_title(f"Federated Learning — Cycle {state.cycle_number} Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"fl_loss_cycle{state.cycle_number}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[FL] Loss plot saved → {path}")

def start_next_round():
    """Kick off a new round: shuffle training order, point to first trainer."""
    state.round_number     += 1
    state.training_order    = list(state.registered_clients.keys())
    state.current_trainer_idx = 0
    state.round_updates     = []
    state.round_losses      = {}
    state.phase             = "training"
    trainer = state.training_order[0]
    print(f"\n[Round {state.round_number}] Starting — training order: {state.training_order}")
    print(f"[Round {state.round_number}] Instructing {trainer} to train")

def aggregate_round():
    """Run FedAvg, check convergence, advance state."""
    print(f"[Round {state.round_number}] Aggregating {len(state.round_updates)} updates …")
    state.global_weights = federated_average(state.round_updates)
    state.phase = "aggregating"

    # Compute average round loss from all clients' reported losses
    all_losses = []
    for losses in state.round_losses.values():
        if losses:
            all_losses.append(losses[-1])   # last epoch loss per client
    avg_loss = float(np.mean(all_losses)) if all_losses else float('inf')
    state.history_loss.append(avg_loss)
    print(f"[Round {state.round_number}] Avg loss = {avg_loss:.5f}")

    save_plots()

    converged, reason = check_convergence()
    if converged or state.round_number >= MAX_ROUNDS:
        reason = reason or f"reached max rounds ({MAX_ROUNDS})"
        print(f"\n✓ FL Converged! Reason: {reason}")
        print(f"  Final global round: {state.round_number}")

        # Verify weights identical across server record (clients will confirm on their end)
        print(f"  [Integrity] Global weights hash: {hash(str(state.global_weights))}")

        # Convert final weights → TFLite. Inference instruction is gated on
        # tflite_ready so clients never get sent to inference mode before
        # the .tflite binary exists and its hash is known.
        build_and_save_tflite_model()

        state.phase         = "converged"
        state.cooldown_start = None   # inference mode indefinitely until next cycle
    else:
        # Start next round immediately
        start_next_round()

# ─── Registration ──────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    client_id = data.get("client_id")
    if not client_id:
        return jsonify({"error": "missing client_id"}), 400

    with lock:
        state.registered_clients[client_id] = time.time()
        n = len(state.registered_clients)
        print(f"[FL] Client registered: {client_id}  (total={n})")

        # Once enough clients register, kick off round 1
        if n >= MIN_CLIENTS and state.phase == "waiting_for_clients":
            start_next_round()

    return jsonify({"status": "registered", "client_count": n})

# ─── Status / instruction endpoint (clients poll this) ────────────────────────

@app.route("/status", methods=["GET"])
def status():
    client_id = request.args.get("client_id", "unknown")
    with lock:
        state.registered_clients[client_id] = time.time()   # heartbeat

        # Check if we have enough clients to start the first round
        if state.phase == "waiting_for_clients" and len(state.registered_clients) >= MIN_CLIENTS:
            start_next_round()

        response = {
            "phase":        state.phase,
            "round":        state.round_number,
            "cycle":        state.cycle_number,
            "total_rounds": len(state.history_loss),
        }

        if state.phase == "waiting_for_clients":
            response["instruction"]   = "wait"
            response["waiting_for"]   = max(0, MIN_CLIENTS - len(state.registered_clients))

        elif state.phase == "training":
            current_trainer = (state.training_order[state.current_trainer_idx]
                               if state.current_trainer_idx < len(state.training_order)
                               else None)
            if client_id == current_trainer:
                response["instruction"] = "train"
                response["weights"]     = state.global_weights
            else:
                response["instruction"] = "wait"
                response["waiting_for_client"] = current_trainer

        elif state.phase == "aggregating":
            response["instruction"] = "wait"

        elif state.phase == "converged":
            if not state.tflite_ready:
                # TFLite conversion not yet complete (or failed) — hold clients
                response["instruction"] = "wait"
                response["waiting_reason"] = "tflite_conversion_pending"
            elif not state.client_model_acks.get(client_id, False):
                # Client hasn't confirmed the TFLite model hash yet — send it
                # the download instruction; client must hit /confirm_model
                # before receiving "inference"
                response["instruction"]  = "download_model"
                response["weights"]      = state.global_weights   # for hand-written fallback
                response["tflite_hash"]  = state.tflite_hash
                response["tflite_size"]  = state.tflite_size_bytes
                response["tflite_url"]   = "/model.tflite"
            else:
                # All checks passed — release inference instruction
                response["instruction"] = "inference"
                response["weights"]     = state.global_weights
                response["tflite_hash"] = state.tflite_hash
                response["message"]     = "FL converged. All clients verified. Run dual inference."
                # Once ALL registered clients have acked, start cooldown
                all_acked = all(state.client_model_acks.get(c, False)
                                for c in state.registered_clients)
                if all_acked and state.cooldown_start is None:
                    state.cooldown_start = time.time()
                    print(f"[FL] All clients confirmed model — cooldown started "
                          f"({INFERENCE_COOLDOWN_S}s until next cycle)")

        elif state.phase == "cooldown":
            elapsed = time.time() - (state.cooldown_start or time.time())
            remaining = max(0, INFERENCE_COOLDOWN_S - elapsed)
            response["instruction"] = "inference"
            response["cooldown_remaining_s"] = int(remaining)
            if remaining <= 0:
                print(f"\n[FL] Cooldown complete — starting cycle {state.cycle_number + 1}")
                state.cycle_number  += 1
                state.plateau_count  = 0
                state.history_loss   = []
                state.global_weights = init_weights()
                state.tflite_ready      = False
                state.tflite_hash       = None
                state.client_model_acks = {}
                state.benchmark_reports = {}
                start_next_round()
                response["instruction"] = "train" if client_id == state.training_order[0] else "wait"

    return jsonify(response)

# ─── Model download (used during training instruction) ────────────────────────

@app.route("/model", methods=["GET"])
def get_model():
    with lock:
        return jsonify({"round": state.round_number, "weights": state.global_weights})

# ─── Weight upload ────────────────────────────────────────────────────────────

@app.route("/update", methods=["POST"])
def receive_update():
    data = request.get_json(force=True)
    if not data or "weights" not in data:
        return jsonify({"error": "missing weights"}), 400

    client_id  = data.get("client_id", "unknown")
    weights    = data["weights"]
    n_samples  = data.get("n_samples", 1)
    epoch_losses = data.get("epoch_losses", [])    # list of per-epoch loss values

    # Validate weight count
    flat = []
    for layer in weights:
        flat.extend(layer if isinstance(layer, list) else [layer])
    if len(flat) != TOTAL_WEIGHTS:
        return jsonify({"error": f"wrong weight count: {len(flat)}"}), 400

    with lock:
        # Verify it's this client's turn
        if state.phase != "training":
            return jsonify({"error": "server not in training phase", "phase": state.phase}), 409

        expected_trainer = (state.training_order[state.current_trainer_idx]
                            if state.current_trainer_idx < len(state.training_order)
                            else None)
        if client_id != expected_trainer:
            return jsonify({"error": f"not your turn — waiting for {expected_trainer}"}), 409

        print(f"[Round {state.round_number}] ✓ Received weights from {client_id} ({n_samples} samples)")
        if epoch_losses:
            print(f"  Epoch losses: {[f'{l:.4f}' for l in epoch_losses]}")

        state.round_updates.append((weights, n_samples))
        state.round_losses[client_id] = epoch_losses

        state.current_trainer_idx += 1

        # Check if all clients in this round have submitted
        if state.current_trainer_idx >= len(state.training_order):
            print(f"[Round {state.round_number}] All clients submitted — running FedAvg …")
            # Run aggregation in a background thread to not block the HTTP response
            threading.Thread(target=aggregate_round, daemon=True).start()
        else:
            next_trainer = state.training_order[state.current_trainer_idx]
            print(f"[Round {state.round_number}] Instructing next trainer: {next_trainer}")

    return jsonify({
        "status":       "received",
        "global_round": state.round_number,
        "next_trainer": (state.training_order[state.current_trainer_idx]
                         if state.current_trainer_idx < len(state.training_order)
                         else "aggregating"),
    })

# ─── Convergence / weight verification endpoint ───────────────────────────────

@app.route("/verify_weights", methods=["POST"])
def verify_weights():
    """Client submits its final weights; server checks they match global weights."""
    data      = request.get_json(force=True)
    client_id = data.get("client_id", "unknown")
    weights   = data.get("weights", [])

    with lock:
        match = weights_are_equal(state.global_weights, weights)
        print(f"[FL] Weight verification — {client_id}: {'✓ MATCH' if match else '✗ MISMATCH'}")

    return jsonify({"match": match, "client_id": client_id})

# ─── TFLite model distribution ─────────────────────────────────────────────────

@app.route("/model.tflite", methods=["GET"])
def get_tflite_model():
    """Serve the converted .tflite binary file for clients to download."""
    with lock:
        if not state.tflite_ready:
            return jsonify({"error": "tflite model not ready"}), 404
    return send_file(TFLITE_MODEL_PATH, mimetype="application/octet-stream")

@app.route("/confirm_model", methods=["POST"])
def confirm_model():
    """
    Client confirms it has downloaded the .tflite file and computed a matching
    SHA256 hash locally. This is the integrity check that ensures server and
    ALL clients are running the exact same model before inference begins.
    """
    data         = request.get_json(force=True)
    client_id    = data.get("client_id", "unknown")
    client_hash  = data.get("tflite_hash", "")

    with lock:
        if not state.tflite_ready:
            return jsonify({"error": "no tflite model on server yet"}), 409

        match = (client_hash == state.tflite_hash)
        state.client_model_acks[client_id] = match

        status_str = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"[TFLite] Model confirmation — {client_id}: {status_str}")
        if not match:
            print(f"  Server hash: {state.tflite_hash}")
            print(f"  Client hash: {client_hash}")

        all_acked = all(state.client_model_acks.get(c, False)
                        for c in state.registered_clients)

    return jsonify({
        "match":             match,
        "all_clients_ready": all_acked,
        "acked_count":       sum(1 for v in state.client_model_acks.values() if v),
        "total_clients":     len(state.registered_clients),
    })

# ─── Inference benchmark reporting ─────────────────────────────────────────────

@app.route("/benchmark_report", methods=["POST"])
def benchmark_report():
    """
    Client reports a comparison of hand-written vs TFLM inference for one
    forward pass: latency (us), memory footprint (bytes), and prediction
    confidence for both methods. Used to build the KPI comparison table.
    """
    data      = request.get_json(force=True)
    client_id = data.get("client_id", "unknown")

    report = {
        "client_id":           client_id,
        "timestamp":           time.time(),
        "handwritten_latency_us": data.get("handwritten_latency_us"),
        "tflm_latency_us":        data.get("tflm_latency_us"),
        "handwritten_ram_bytes":  data.get("handwritten_ram_bytes"),
        "tflm_arena_bytes":       data.get("tflm_arena_bytes"),
        "handwritten_confidence": data.get("handwritten_confidence"),
        "tflm_confidence":        data.get("tflm_confidence"),
        "predictions_match":      data.get("predictions_match"),
        "predicted_class":        data.get("predicted_class"),
    }

    with lock:
        state.benchmark_reports[client_id] = report

    print(f"[Benchmark] {client_id} — "
          f"HW: {report['handwritten_latency_us']}us / "
          f"TFLM: {report['tflm_latency_us']}us  "
          f"(match={report['predictions_match']})")

    return jsonify({"status": "received"})

@app.route("/benchmark_summary", methods=["GET"])
def benchmark_summary():
    """Aggregate KPI comparison across all clients' latest reports."""
    with lock:
        reports = list(state.benchmark_reports.values())

    if not reports:
        return jsonify({"message": "no benchmark reports yet", "reports": []})

    def safe_mean(key):
        vals = [r[key] for r in reports if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "client_count":              len(reports),
        "avg_handwritten_latency_us": safe_mean("handwritten_latency_us"),
        "avg_tflm_latency_us":        safe_mean("tflm_latency_us"),
        "avg_handwritten_ram_bytes":  safe_mean("handwritten_ram_bytes"),
        "avg_tflm_arena_bytes":       safe_mean("tflm_arena_bytes"),
        "avg_handwritten_confidence": safe_mean("handwritten_confidence"),
        "avg_tflm_confidence":        safe_mean("tflm_confidence"),
        "agreement_rate":             safe_mean("predictions_match"),
        "tflite_model_size_bytes":    state.tflite_size_bytes,
        "reports":                    reports,
    }

    if summary["avg_handwritten_latency_us"] and summary["avg_tflm_latency_us"]:
        speedup = summary["avg_handwritten_latency_us"] / summary["avg_tflm_latency_us"]
        summary["tflm_speedup_factor"] = round(speedup, 2)

    return jsonify(summary)

# ─── Diagnostics ──────────────────────────────────────────────────────────────

@app.route("/dashboard", methods=["GET"])
def dashboard():
    with lock:
        return jsonify({
            "phase":              state.phase,
            "cycle":              state.cycle_number,
            "round":              state.round_number,
            "registered_clients": list(state.registered_clients.keys()),
            "training_order":     state.training_order,
            "current_trainer":    (state.training_order[state.current_trainer_idx]
                                   if state.phase == "training" and
                                      state.current_trainer_idx < len(state.training_order)
                                   else None),
            "history_loss":       state.history_loss,
            "plateau_count":      state.plateau_count,
            "min_clients":        MIN_CLIENTS,
            "total_weights":      TOTAL_WEIGHTS,
            "tflite_ready":       state.tflite_ready,
            "tflite_hash":        state.tflite_hash,
            "tflite_size_bytes":  state.tflite_size_bytes,
            "client_model_acks":  state.client_model_acks,
        })

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Federated Learning Orchestration Server")
    print(f"  Model:          {INPUT_DIM}→{HIDDEN1}→{HIDDEN2}→{OUTPUT_DIM}  ({TOTAL_WEIGHTS} weights)")
    print(f"  Min clients:    {MIN_CLIENTS}  (configurable)")
    print(f"  Convergence:    loss < {CONVERGENCE_LOSS}  OR  plateau × {CONVERGENCE_PLATEAU}")
    print(f"  Max rounds:     {MAX_ROUNDS}")
    print(f"  Inference gap:  {INFERENCE_COOLDOWN_S}s ({INFERENCE_COOLDOWN_S//60} min)")
    print(f"  TFLite:         int8 quantised, served at /model.tflite on convergence")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
