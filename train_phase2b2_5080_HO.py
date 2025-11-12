#! python
"""
train_phase2b2_HO.py
2bANN.2 Hybrid Architecture - Heating Oil (HO) dataset
Audit-safe training script with explicit train/val/test splits
"""

import os
import hashlib
import datetime
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Data loading ---
# BROKEN: def load_ho_csv(path):
    colnames = ["Date", "Open", "High", "Low", "Close"]
    df = pd.read_csv(
        path,
        sep=None,              # auto-detect delimiter
        engine="python",       # required for sep=None
        header=0,
        names=colnames,
        parse_dates=['Date']
    )
    dates = df['Date'].values
    X = df[['Open','High','Low']].values.astype(float)
    y = df['Close'].values.astype(float)
    print(f"{os.path.basename(path)} -> dates: {dates.shape}, X: {X.shape}, y: {y.shape}")
    return dates, X, y

# --- Evaluation helpers ---
# BROKEN: def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    return mae, rmse, r2

# BROKEN: def plot_predictions(dates, y_true, y_pred, title="ANN Predictions vs Actual"):
# BROKEN:     try:
        import matplotlib.pyplot as plt
# BROKEN:     except ImportError:
        print("[SKIPPED] matplotlib not installed, cannot plot.")
        return

    plt.figure(figsize=(12,6))
    plt.plot(dates, y_true, label="Actual Close", color="black", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted Close", color="red", linestyle="--", linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- Provenance logging ---
# BROKEN: def hash_file(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# BROKEN: def log_run(meta):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"RUNLOG_2bANN2_HO_{ts}.json"
# BROKEN:     with open(log_name, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[LOGGED] {log_name}")

# --- Dataset paths ---
train_path = r"C:\Users\loweb\AI_Financial_Sims\Prototypes\Prototype1_Regression\HO\hoxnc_training.csv"
val_path   = r"C:\Users\loweb\AI_Financial_Sims\Prototypes\Prototype1_Regression\HO\hoxnc_validation.csv"
test_path  = r"C:\Users\loweb\AI_Financial_Sims\Prototypes\Prototype1_Regression\HO\hoxnc_testing.csv"

# --- Load datasets ---
dates_train, X_train, y_train = load_ho_csv(train_path)
dates_val,   X_val,   y_val   = load_ho_csv(val_path)
dates_test,  X_test,  y_test  = load_ho_csv(test_path)

# --- Scaling ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# --- Hybrid ANN architecture ---
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  # regression output
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# --- Training ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=2
)

# --- Evaluation ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# --- Predictions + metrics ---
y_pred = model.predict(X_test).flatten()
evaluate_predictions(y_test, y_pred)
# plot_predictions(dates_test, y_test, y_pred)  # safe to call if matplotlib is installed

# --- Save model ---
MODEL_PATH = "2bANN2_HO_model.keras"
model.save(MODEL_PATH)
print(f"[SAVED] {MODEL_PATH}")

# --- Log metadata ---
meta = {
    "script": "train_phase2b2_HO.py",
    "train_file": train_path,
    "val_file": val_path,
    "test_file": test_path,
    "train_hash": hash_file(train_path),
    "val_hash": hash_file(val_path),
    "test_hash": hash_file(test_path),
    "model_file": MODEL_PATH,
    "model_hash": hash_file(MODEL_PATH),
    "test_loss": float(loss),
    "test_mae": float(mae),
    "timestamp": datetime.datetime.now().isoformat()
}
log_run(meta)
