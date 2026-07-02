"""
Federated Learning Orchestration Server
- Manages round lifecycle: init → sequential client training → FedAvg → convergence check
- Clients poll /status to receive instructions
- Sequential training: one client trains at a time
- Convergence: loss below threshold OR plateau detection
- After convergence: inference mode, 15-min cooldown, then next FL cycle
"""

import json
import time
import threading
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
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
            response["instruction"] = "inference"
            response["weights"]     = state.global_weights
            response["message"]     = "FL converged. Switch to inference mode."
            # After clients acknowledge, start cooldown
            if state.cooldown_start is None:
                state.cooldown_start = time.time()
                print(f"[FL] Cooldown started — next cycle in {INFERENCE_COOLDOWN_S}s")

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
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
