# fishhead_ho_audit_v5_2_20251029.py
# Purpose: Audit-safe Fishhead ANN on Heating Oil (HO) daily data with closure-marked outputs.
# - Input: hoxnc_full.csv (concatenated HO dataset)
# - Target: ΔClose[t+1] = Close[t+1] - Close[t] (time-shifted, no leakage)
# - Features: Demeaned Close window (size=128)
# - Splits: Contiguous train/val/test with purge+embargo
# - Metrics: MAE, RMSE, R², MAPE vs Naïve (ΔClose=0), plus quantile coverage, Brier, abstention
# - Artifacts: model .pt, metrics CSV, run log JSON, plots; metrics printed at end
# Version: v5.2
# Date: 29 October 2025

import os, json, random, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, brier_score_loss
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Config
# ------------------------------
SEED = 5080
WINDOW = 128
EVENT_REL_THRESHOLD = 0.02
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.6, 0.2, 0.2
PURGE_STEPS = 5
EMBARGO_STEPS = 5

CSV_PATH = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_full.csv"
OUTDIR = "fishhead_ho_audit_outputs_v5_2"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------
# Utilities
# ------------------------------
# BROKEN: def set_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# BROKEN: def load_ho_csv(path=CSV_PATH):
    df = pd.read_csv(path)
    df = df[['Date','Close']].dropna().copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# BROKEN: def make_dataset_from_close(close: np.ndarray, window=WINDOW, event_thresh=EVENT_REL_THRESHOLD):
    X, y, ev, pc_forecast = [], [], [], []
# BROKEN:     for i in range(window, len(close)-1):
        w = close[i-window:i]
        X.append((w - w.mean()).astype(np.float32))
        d_close = close[i+1] - close[i]
        y.append(np.float32(d_close))
        rel_move = abs(d_close / close[i])
        ev.append(np.float32(1 if rel_move >= event_thresh else 0))
        pc_forecast.append(np.float32(0.0))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1,1)
    ev = np.array(ev, dtype=np.float32).reshape(-1,1)
    pc_forecast = np.array(pc_forecast, dtype=np.float32).reshape(-1,1)
    return X, y, ev, pc_forecast

# BROKEN: def contiguous_splits(n, fracs=(TRAIN_FRAC, VAL_FRAC, TEST_FRAC), purge=PURGE_STEPS, embargo=EMBARGO_STEPS):
    t = int(n * fracs[0]); v = int(n * fracs[1])
    train = (0, max(0, t - purge))
    val_start = min(n, t + embargo)
    val = (val_start, min(n, val_start + v - purge))
    test_start = min(n, val[1] + embargo)
    test = (test_start, n)
    return train, val, test

# ------------------------------
# Model
# ------------------------------
# BROKEN: class FishheadANN(nn.Module):
# BROKEN:     def __init__(self, input_dim=WINDOW, hidden=96, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.dp = nn.Dropout(dropout)
        self.resid = nn.Linear(hidden, 1)
        self.quant = nn.Linear(hidden, 3)
        self.event = nn.Linear(hidden, 1)
        self.gate  = nn.Linear(hidden, 1)
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

# ------------------------------
# Metrics helpers
# ------------------------------
# BROKEN: def coverage_rate(q_lo, q_hi, y_true):
    inside = (y_true >= q_lo) & (y_true <= q_hi)
    return float(np.mean(inside))

# BROKEN: def mape_safe(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred)) / denom)) * 100.0

# ------------------------------
# Run
# ------------------------------
# BROKEN: def run():
    set_seeds(SEED)
    df = load_ho_csv(CSV_PATH)
    close = df['Close'].values.astype(np.float32)

    X, y, ev, naive_dclose = make_dataset_from_close(close, WINDOW, EVENT_REL_THRESHOLD)
    n = len(X)
# BROKEN:     (t0,t1), (v0,v1), (s0,s1) = contiguous_splits(n)

    X_t = torch.from_numpy(X[t0:t1]); y_t = torch.from_numpy(y[t0:t1]); ev_t = torch.from_numpy(ev[t0:t1])
    X_v = torch.from_numpy(X[v0:v1]); y_v = torch.from_numpy(y[v0:v1]); ev_v = torch.from_numpy(ev[v0:v1])
    X_s = torch.from_numpy(X[s0:s1]); y_s = torch.from_numpy(y[s0:s1]); ev_s = torch.from_numpy(ev[s0:s1])

    model = FishheadANN(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 60
# BROKEN:     for epoch in range(epochs):
        model.train(); opt.zero_grad()
        out = model(X_t)
        loss = (
            F.mse_loss(out["residual"], y_t) +
            quantile_loss(out["quantiles"], y_t) +
            F.binary_cross_entropy(out["event_prob"], ev_t) +
            0.01 * out["gate"].mean()
        )
        loss.backward(); opt.step()
# BROKEN:         if (epoch+1) % 10 == 0:
            print(f"[Fishhead] Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    model.eval()
# BROKEN:     with torch.no_grad():
        ov = model(X_v); os_ = model(X_s)

    q_v = ov["quantiles"].numpy(); p_v = ov["event_prob"].numpy().flatten(); g_v = ov["gate"].numpy().flatten()
    y_val = y_v.numpy().flatten()
    q_s = os_["quantiles"].numpy(); p_s = os_["event_prob"].numpy().flatten(); g_s = os_["gate"].numpy().flatten()
    y_test = y_s.numpy().flatten()

    naive_v = np.zeros_like(y_val); naive_s = np.zeros_like(y_test)
    yhat_v = q_v[:,1]; yhat_s = q_s[:,1]

    # Validation metrics
    rmse_fish_v = float(np.sqrt(mean_squared_error(y_val, yhat_v)))
    mae_fish_v  = float(mean_absolute_error(y_val, yhat_v))
    r2_fish_v   = float(r2_score(y_val, yhat_v))
    mape_fish_v = mape_safe(y_val, yhat_v)

    rmse_naive_v = float(np.sqrt(mean_squared_error(y_val, naive_v)))
    mae_naive_v  = float(mean_absolute_error(y_val, naive_v))
    r2_naive_v   = float(r2_score(y_val, naive_v))
    mape_naive_v = mape_safe(y_val, naive_v)

    cov_v = coverage_rate(q_v[:,0], q_v[:,2], y_val)
    idx_val_start = WINDOW + v0
    prev_close_val = close[idx_val_start:idx_val_start+len(y_val)]
    ev_truth_v = (np.abs(y_val / prev_close_val) >= EVENT_REL_THRESHOLD).astype(int)
    brier_v = float(brier_score_loss(ev_truth_v, p_v))
    abstain_v = float(np.mean(g_v < 0.3))

    # Test metrics
    rmse_fish_s = float(np.sqrt(mean_squared_error(y_test, yhat_s)))
    mae_fish_s  = float(mean_absolute_error(y_test, yhat_s))
    r2_fish_s   = float(r2_score(y_test, yhat_s))
    mape_fish_s = mape_safe(y_test, yhat_s)

    rmse_naive_s = float(np.sqrt(mean_squared_error(y_test, naive_s)))
    mae_naive_s  = float(mean_absolute_error(y_test, naive_s))
    r2_naive_s   = float(r2_score(y_test, naive_s))
    mape_naive_s = mape_safe(y_test, naive_s)

    cov_s = coverage_rate(q_s[:,0], q_s[:,2], y_test)
    idx_test_start = WINDOW + s0
    prev_close_test = close[idx_test_start:idx_test_start+len(y_test)]
    ev_truth_s = (np.abs(y_test / prev_close_test) >= EVENT_REL_THRESHOLD).astype(int)
    brier_s = float(brier_score_loss(ev_truth_s, p_s))
    abstain_s = float(np.mean(g_s < 0.3))

    metrics_row = {
        "rmse_fish_val": rmse_fish_v,
        "mae_fish_val": mae_fish_v,
        "r2_fish_val": r2_fish_v,
        "mape_fish_val_pct": mape_fish_v,
        "rmse_naive_val": rmse_naive_v,
        "mae_naive_val": mae_naive_v,
        "r2_naive_val": r2_naive_v,
        "mape_naive_val_pct": mape_naive_v,
        "coverage_val_q10_q90": cov_v,
        "brier_val": brier_v,
        "abstention_val_gate<0.3": abstain_v,
        "rmse_fish_test": rmse_fish_s,
        "mae_fish_test": mae_fish_s,
        "r2_fish_test": r2_fish_s,
        "mape_fish_test_pct": mape_fish_s,
        "rmse_naive_test": rmse_naive_s,
        "mae_naive_test": mae_naive_s,
        "r2_naive_test": r2_naive_s,
        "mape_naive_test_pct": mape_naive_s,
        "coverage_test_q10_q90": cov_s,
        "brier_test": brier_s,
        "abstention_test_gate<0.3": abstain_s,
        "n_train": (t1 - t0),
        "n_val": (v1 - v0),
        "n_test": (s1 - s0),
        "window": WINDOW,
        "event_rel_threshold": EVENT_REL_THRESHOLD,
        "purge_steps": PURGE_STEPS,
        "embargo_steps": EMBARGO_STEPS
    }

    metrics_df = pd.DataFrame([metrics_row])

    # Save artifacts
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = os.path.join(OUTDIR, f"fishhead_metrics_v5_2_{ts}.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    model_path = os.path.join(OUTDIR, f"fishhead_model_v5_2_{ts}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": X.shape[1],
                "hidden": 96,
                "dropout": 0.1,
                "window": WINDOW,
                "seed": SEED
            }
        },
        model_path
    )

    runlog = {
        "timestamp": ts,
        "csv_path": CSV_PATH,
        "outdir": OUTDIR,
        "splits": {"train": [t0, t1], "val": [v0, v1], "test": [s0, s1]},
        "metrics_csv": metrics_csv,
        "model_path": model_path,
        "metrics": metrics_row
    }
    runlog_path = os.path.join(OUTDIR, f"fishhead_runlog_v5_2_{ts}.json")
# BROKEN:     with open(runlog_path, "w") as f:
        json.dump(runlog, f, indent=2)

    # Plots (validation)
    plt.figure(figsize=(10,5))
    plt.plot(y_val, label="True ΔClose (val)", color="black", alpha=0.7)
    plt.plot(yhat_v, label="Fishhead q50", color="blue")
    plt.plot(naive_v, label="Naïve ΔClose=0", color="red", alpha=0.6)
    plt.legend(); plt.title("Val: True vs Fishhead q50 vs Naïve ΔClose")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"fishhead_val_true_q50_naive_v5_2_{ts}.png")); plt.close()

    plt.figure(figsize=(10,5))
    plt.fill_between(range(len(y_val)), q_v[:,0], q_v[:,2], color="lightblue", alpha=0.4, label="q10–q90")
    plt.plot(y_val, color="black", alpha=0.7, label="True ΔClose (val)")
    plt.legend(); plt.title("Val: Quantile coverage")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"fishhead_val_quantile_coverage_v5_2_{ts}.png")); plt.close()

    prob_true_v, prob_pred_v = calibration_curve(ev_truth_v, p_v, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred_v, prob_true_v, marker="o", label="Val")
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfect")
    plt.xlabel("Predicted prob"); plt.ylabel("Observed freq"); plt.title("Val: Event calibration")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"fishhead_val_event_calibration_v5_2_{ts}.png")); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(g_v, label="Gate (val)", color="blue")
    plt.axhline(0.3, color="red", linestyle="--", label="Threshold 0.3")
    plt.title("Val: Gate/Abstention"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"fishhead_val_gate_v5_2_{ts}.png")); plt.close()

    # Console printout
    print("\n=== Fishhead HO Audit v5.2 Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\nArtifacts:")
    print(f"- Metrics CSV: {metrics_csv}")
    print(f"- Model:       {model_path}")
    print(f"- Run log:     {runlog_path}")
    print("\nHO audit run complete.")

# BROKEN: if __name__ == "__main__":
    run()
