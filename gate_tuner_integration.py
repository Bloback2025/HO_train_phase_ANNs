# gate_tuner_integration.py
# Purpose: Integrate Fishhead outputs with a GateTuner ANN that learns
#          dynamic abstention thresholds based on quantile spread,
#          residual error, and raw gate score. Demonstrates an ANN
#          tuning another ANN for decision utility.
# Date: 26 October 2025
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- GateTuner ANN ---
# BROKEN: class GateTuner(nn.Module):
# BROKEN:     def __init__(self, input_dim=3, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
# BROKEN:     def forward(self, x):
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))  # tuned threshold in [0,1]

# --- Training harness ---
# BROKEN: def train_gate_tuner(q_preds, gate_scores, y_true, event_probs, epochs=50):
    """
    q_preds: np.array [N,3] quantiles from Fishhead
    gate_scores: np.array [N] raw gate outputs
    y_true: np.array [N] true ΔClose
    event_probs: np.array [N] predicted event probabilities
    """
    # Features: quantile spread, gate score, abs residual error
    spread = q_preds[:,2] - q_preds[:,0]
    abs_err = np.abs(y_true - q_preds[:,1])
    feats = np.stack([spread, gate_scores, abs_err], axis=1).astype(np.float32)

    X = torch.from_numpy(feats)
    # Target: whether to act (event probability > 0.5)
    y = torch.from_numpy((event_probs > 0.5).astype(np.float32))

    model = GateTuner(input_dim=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# BROKEN:     for epoch in range(epochs):
        opt.zero_grad()
        thr = model(X).squeeze()  # tuned thresholds
        # Decision: act if event_prob > thr
        act = (torch.from_numpy(event_probs).float() > thr).float()
        # Utility: reward correct acts, penalize wrong ones
        loss = F.binary_cross_entropy(act, y)
        loss.backward(); opt.step()
# BROKEN:         if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: tuner loss={loss.item():.4f}")

    return model

# Example usage (after running Fishhead and collecting outputs):
# tuner = train_gate_tuner(q_preds, gate_scores, y_true, event_probs)
# tuned_thresholds = tuner(torch.from_numpy(features)).detach().numpy()
