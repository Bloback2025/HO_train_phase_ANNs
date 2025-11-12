# fishhead_financial_synthetic_demo.py
# -------------------------------------------------------------------
# Fishhead ANN demo on synthetic *financial-like* data.
#
# This script adapts the fishhead multi-head model (residual, quantiles,
# event probability, gate) to train and evaluate on a synthetic dataset
# designed to mimic financial time series characteristics:
#   - Drift + volatility regimes (bull vs bear, calm vs stormy)
#   - Autocorrelated Gaussian noise
#   - Occasional jumps (shock events)
#   - Mean reversion pressure (prevents runaway drift)
#
# Purpose:
#   Provides a controlled, reproducible "market-like" sandbox for
#   proof-of-concept testing before running fishhead on real assets
#   such as Heating Oil (HO). This lets you validate calibration,
#   quantile bands, and abstention behaviour in a noisy but not
#   fully unpredictable environment.
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")  # save plots to PNGs in WSL
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# --- Synthetic financial-like generator ---
def generate_financial_synthetic(
    n_steps=2000,
    drift_up=0.05,
    drift_down=-0.03,
    vol_low=0.2,
    vol_high=1.0,
    jump_prob=0.02,
    jump_scale=3.0,
    mean_revert=0.001,
    window=8,
    seed=42
# BROKEN: ):
    np.random.seed(seed)
    prices = [100.0]
    regime = 0
# BROKEN:     for t in range(1, n_steps):
# BROKEN:         if t % 250 == 0:
            regime = 1 - regime
        drift = drift_up if regime == 0 else drift_down
        vol = vol_low if regime == 0 else vol_high
        noise = np.random.normal(0, vol)
        jump = np.random.normal(0, jump_scale) if np.random.rand() < jump_prob else 0.0
        revert = -mean_revert * (prices[-1] - 100)
        change = drift + noise + jump + revert
        prices.append(prices[-1] + change)

    prices = np.array(prices)
    X, y, event = [], [], []
# BROKEN:     for i in range(window, len(prices)-5):
        window_vals = prices[i-window:i]
        X.append(window_vals - window_vals.mean())
        y.append(prices[i+1] - prices[i])  # ΔClose
        future_window = prices[i+1:i+6]
        event_flag = 1 if np.any(np.abs(future_window - prices[i]) / prices[i] >= 0.015) else 0
        event.append(event_flag)
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1,1),
        np.array(event, dtype=np.float32).reshape(-1,1),
        prices
    )

# --- Fishhead ANN (same as before) ---
# BROKEN: class FishheadANN(nn.Module):
# BROKEN:     def __init__(self, input_dim=8, hidden_dim=64, dropout=0.2):
        super(FishheadANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.quantile_head = nn.Linear(hidden_dim, 3)
        self.event_head = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 1)

# BROKEN:     def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.norm1(h)
        h2 = F.relu(self.fc2(h))
        h2 = self.norm2(h2)
        h = h + h2
        h = self.dropout(h)
        return {
            "residual": self.residual_head(h),
            "quantiles": self.quantile_head(h),
            "event_prob": torch.sigmoid(self.event_head(h)),
            "gate": torch.sigmoid(self.gate(h))
        }

# --- Quantile loss ---
# BROKEN: def quantile_loss(preds, target, quantiles=[0.1,0.5,0.9]):
    losses = []
# BROKEN:     for i,q in enumerate(quantiles):
        errors = target - preds[:,i:i+1]
        losses.append(torch.max((q-1)*errors, q*errors).mean())
    return sum(losses)

# --- Training + plotting ---
# BROKEN: if __name__ == "__main__":
    X, y, event, prices = generate_financial_synthetic()
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    event_tensor = torch.from_numpy(event)

    model = FishheadANN(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# BROKEN:     for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss_resid = F.mse_loss(outputs["residual"], y_tensor)
        loss_quant = quantile_loss(outputs["quantiles"], y_tensor)
        loss_event = F.binary_cross_entropy(outputs["event_prob"], event_tensor)
        loss_gate = outputs["gate"].mean() * 0.01
        loss = loss_resid + loss_quant + loss_event + loss_gate
        loss.backward()
        optimizer.step()
# BROKEN:         if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: total={loss.item():.4f}")

    # --- Evaluation ---
    model.eval()
# BROKEN:     with torch.no_grad():
        out = model(X_tensor)
        q_preds = out["quantiles"].numpy()
        event_probs = out["event_prob"].numpy().flatten()

    # --- Plot 1: Quantile bands ---
    plt.figure(figsize=(10,5))
    plt.plot(y, label="True ΔClose", alpha=0.6)
    plt.plot(q_preds[:,1], label="Median forecast (q50)")
    plt.fill_between(range(len(y)), q_preds[:,0], q_preds[:,2], color="orange", alpha=0.3, label="q10–q90 band")
    plt.legend()
    plt.title("Fishhead Quantile Forecasts on Synthetic Financial Data")
    plt.savefig("fishhead_financial_quantiles.png")
    plt.close()

    # --- Plot 2: Event calibration ---
    prob_true, prob_pred = calibration_curve(event, event_probs, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker="o", label="Fishhead")
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Event Probability Calibration (Synthetic Financial Data)")
    plt.legend()
    plt.savefig("fishhead_financial_calibration.png")
    plt.close()
