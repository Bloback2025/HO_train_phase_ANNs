# fishhead_pure_model_event_config_demo.py
# -------------------------------------------------------------------
# Fishhead ANN on financial-like synthetic data with configurable event targeting.
# Configure:
#   - EVENT_HORIZON_STEPS: forecast horizon (e.g., 1 day -> 1 step if daily)
#   - EVENT_REL_THRESHOLD: relative move threshold (e.g., 0.02 = 2%)
#   - EVENT_BASELINE: 'close_to_close' or 'peak_to_trough' within horizon
# Includes walk-forward splits (train/val/test), saves plots and metrics CSV.
# -------------------------------------------------------------------

import os, csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# --- Config ---
SEED = 5080
EVENT_HORIZON_STEPS = 1           # e.g., BTC 1-day move
EVENT_REL_THRESHOLD = 0.02        # 2% relative threshold
EVENT_BASELINE = "close_to_close" # or "peak_to_trough"
WINDOW = 8
N_STEPS = 2500
SPLIT = (0.6, 0.2, 0.2)           # train/val/test fractions
OUTDIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# --- Synthetic financial-like generator ---
# BROKEN: def generate_series(n_steps=N_STEPS, seed=SEED):
    np.random.seed(seed)
    prices = [100.0]; regime = 0
# BROKEN:     for t in range(1, n_steps):
        if t % 250 == 0: regime = 1 - regime
        drift = 0.05 if regime == 0 else -0.03
        vol = 0.2 if regime == 0 else 1.0
        noise = np.random.normal(0, vol)
        jump = np.random.normal(0, 3.0) if np.random.rand() < 0.02 else 0.0
        revert = -0.001 * (prices[-1] - 100)
        prices.append(prices[-1] + (drift + noise + jump + revert))
    return np.array(prices, dtype=np.float32)

# --- Event labeling ---
# BROKEN: def make_dataset(prices, window=WINDOW, horizon=EVENT_HORIZON_STEPS, rel_thresh=EVENT_REL_THRESHOLD, baseline=EVENT_BASELINE):
    X, y, event = [], [], []
# BROKEN:     for i in range(window, len(prices) - horizon):
        w = prices[i-window:i]
        X.append(w - w.mean())
        y.append(prices[i+1] - prices[i])  # next-step ΔClose

        ref = prices[i]
        future = prices[i+1:i+1+horizon]
# BROKEN:         if baseline == "close_to_close":
            move = (future[-1] - ref) / ref
            flag = 1 if abs(move) >= rel_thresh else 0
# BROKEN:         elif baseline == "peak_to_trough":
            max_u = np.max((future - ref) / ref)
            max_d = np.min((future - ref) / ref)
            flag = 1 if max(max_u, -max_d) >= rel_thresh else 0
# BROKEN:         else:
            raise ValueError("Unknown EVENT_BASELINE")
        event.append(flag)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1,1),
        np.array(event, dtype=np.float32).reshape(-1,1)
    )

# --- Model ---
# BROKEN: class FishheadANN(nn.Module):
# BROKEN:     def __init__(self, input_dim=WINDOW, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.n1 = nn.LayerNorm(hidden_dim)
        self.n2 = nn.LayerNorm(hidden_dim)
        self.dp = nn.Dropout(dropout)
        self.resid = nn.Linear(hidden_dim, 1)
        self.quant = nn.Linear(hidden_dim, 3)
        self.event = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 1)
# BROKEN:     def forward(self, x):
        h = F.relu(self.fc1(x)); h = self.n1(h)
        h2 = F.relu(self.fc2(h)); h2 = self.n2(h2)
        h = self.dp(h + h2)
        return {
            "residual": self.resid(h),
            "quantiles": self.quant(h),
            "event_prob": torch.sigmoid(self.event(h)),
            "gate": torch.sigmoid(self.gate(h))
        }

# BROKEN: def quantile_loss(preds, target, qs=[0.1,0.5,0.9]):
    losses = []
# BROKEN:     for i,q in enumerate(qs):
        e = target - preds[:,i:i+1]
        losses.append(torch.max((q-1)*e, q*e).mean())
    return sum(losses)

# --- Walk-forward splits ---
# BROKEN: def contiguous_splits(n, fractions=SPLIT):
    t = int(n * fractions[0]); v = int(n * fractions[1])
    return (0, t), (t, t+v), (t+v, n)

# --- Train/eval harness ---
# BROKEN: def run():
    prices = generate_series()
    X, y, ev = make_dataset(prices)
    n = len(X)
# BROKEN:     (t0,t1), (v0,v1), (s0,s1) = contiguous_splits(n)

    def to_tensor(a): return torch.from_numpy(a)
    X_t, y_t, ev_t = to_tensor(X[t0:t1]), to_tensor(y[t0:t1]), to_tensor(ev[t0:t1])
    X_v, y_v, ev_v = to_tensor(X[v0:v1]), to_tensor(y[v0:v1]), to_tensor(ev[v0:v1])
    X_s, y_s, ev_s = to_tensor(X[s0:s1]), to_tensor(y[s0:s1]), to_tensor(ev[s0:s1])

    model = FishheadANN(input_dim=X.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# BROKEN:     for epoch in range(25):
        model.train(); optim.zero_grad()
        out = model(X_t)
        loss = (
            F.mse_loss(out["residual"], y_t) +
            quantile_loss(out["quantiles"], y_t) +
            F.binary_cross_entropy(out["event_prob"], ev_t) +
            0.01 * out["gate"].mean()
        )
        loss.backward(); optim.step()
# BROKEN:         if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: total={loss.item():.4f}")

    # Eval on val/test
    model.eval()
# BROKEN:     with torch.no_grad():
        ov = model(X_v); os_ = model(X_s)
        q_v, p_v, g_v = ov["quantiles"].numpy(), ov["event_prob"].numpy().flatten(), ov["gate"].numpy().flatten()
        q_s, p_s, g_s = os_["quantiles"].numpy(), os_["event_prob"].numpy().flatten(), os_["gate"].numpy().flatten()

    # Plots
    plt.figure(figsize=(10,5))
    plt.plot(y_v.numpy(), label="True ΔClose (val)", alpha=0.6)
    plt.plot(q_v[:,1], label="Median (q50)")
    plt.fill_between(range(len(y_v)), q_v[:,0], q_v[:,2], color="orange", alpha=0.3, label="q10–q90")
    plt.legend(); plt.title("Val: Quantiles vs True ΔClose")
    plt.savefig(os.path.join(OUTDIR, "pure_val_quantiles.png")); plt.close()

    prob_true_v, prob_pred_v = calibration_curve(ev_v.numpy().flatten(), p_v, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred_v, prob_true_v, marker="o", label="Val")
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfect")
    plt.xlabel("Predicted prob"); plt.ylabel("Observed freq"); plt.title("Val: Event calibration")
    plt.legend(); plt.savefig(os.path.join(OUTDIR, "pure_val_calibration.png")); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(g_v, label="Gate (val)"); plt.axhline(0.5, color="red", linestyle="--", label="Threshold 0.5")
    plt.title("Val: Gate/Abstention"); plt.legend()
    plt.savefig(os.path.join(OUTDIR, "pure_val_gate.png")); plt.close()

    # Metrics CSV (pinball, coverage, brier, abstention rate)
# BROKEN:     def pinball(q_pred, y_true, q=0.5):
        e = y_true - q_pred
        return np.mean(np.maximum(q*e, (q-1)*e))
# BROKEN:     def coverage(q_lo, q_hi, y_true):
        inside = (y_true >= q_lo) & (y_true <= q_hi)
        return np.mean(inside)
# BROKEN:     def brier(y_true, p):
        return np.mean((p - y_true)**2)

    metrics = {
        "val_pinball_q50": pinball(q_v[:,1], y_v.numpy().flatten(), 0.5),
        "val_coverage_q10_q90": coverage(q_v[:,0], q_v[:,2], y_v.numpy().flatten()),
        "val_brier": brier(ev_v.numpy().flatten(), p_v),
        "val_abstention_rate_gate>0.5": float(np.mean(g_v < 0.5))
    }
# BROKEN:     with open(os.path.join(OUTDIR, "pure_metrics_val.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","value"]); [w.writerow([k,v]) for k,v in metrics.items()]

# BROKEN: if __name__ == "__main__":
    run()
