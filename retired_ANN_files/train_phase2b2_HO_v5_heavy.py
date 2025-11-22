"""
train_phase2b2_HO_v5_heavy.py
Heavy ANN forecast run on HO data
"""

import os, sys, json, datetime, hashlib, platform
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, r2_score

# --- Environment control ---
SEED = 5080
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.random.set_seed(SEED)
np.random.seed(SEED)

# --- Mixed precision + XLA ---
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# --- Paths bootstrap (your existing module) ---
from bootstrap_ho_paths_and_patch import BASE_DIR, train_path, val_path, test_path

# --- Utility: hash a file ---
# BROKEN: def sha256(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Load dataset ---
# BROKEN: def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.sort_values("Date").reset_index(drop=True)
    X = df[["Open","High","Low"]].astype(float).values
    y = df["Close"].astype(float).values
    dates = df["Date"].values
    return dates, X, y

dates_train, X_train, y_train = load_csv(train_path)
dates_val,   X_val,   y_val   = load_csv(val_path)
dates_test,  X_test,  y_test  = load_csv(test_path)

# --- Forecast shift: predict Close[t+1] from OH(L)[t] ---
# BROKEN: def shift_forward(X, y):
    return X[:-1], y[1:]
X_train_s, y_train_s = shift_forward(X_train, y_train)
X_val_s,   y_val_s   = shift_forward(X_val,   y_val)
X_test_s,  y_test_s  = shift_forward(X_test,  y_test)

# --- Normalisation (fit on train only, apply to all) ---
mu = X_train_s.mean(axis=0); sigma = X_train_s.std(axis=0) + 1e-9
X_train_n = (X_train_s - mu) / sigma
X_val_n   = (X_val_s   - mu) / sigma
X_test_n  = (X_test_s  - mu) / sigma

# --- Model: deep MLP with regularisation ---
# BROKEN: def build_model(input_dim=3):
    reg = keras.regularizers.l2(1e-4)
    return keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(1, dtype="float32")
    ])

model = build_model()
opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])

# --- Callbacks ---
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(BASE_DIR, f"tb_heavy_v5_{ts}")
ckpt_path = os.path.join(BASE_DIR, f"ckpt_v5_best_{ts}.keras")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir=log_dir)
]

# --- Train ---
history = model.fit(
    X_train_n, y_train_s,
    validation_data=(X_val_n, y_val_s),
    epochs=200,
    batch_size=1024,
    callbacks=callbacks,
    verbose=2
)

# --- Evaluate ANN forecast ---
test_loss, test_mae = model.evaluate(X_test_n, y_test_s, verbose=0)
y_pred = model.predict(X_test_n, verbose=0).reshape(-1)

r2_ann   = r2_score(y_test_s, y_pred)
rmse_ann = np.sqrt(((y_test_s - y_pred)**2).mean())
mape_ann = (np.abs((y_test_s - y_pred) / np.maximum(1e-9, np.abs(y_test_s)))).mean()

# --- Naïve baseline (persistence) ---
y_true  = y_test_s[1:]      # tomorrow's closes
y_naive = y_test_s[:-1]     # today's closes
y_pred_aligned = y_pred[1:] # drop first ANN pred to align

mae_naive  = mean_absolute_error(y_true, y_naive)
rmse_naive = np.sqrt(((y_true - y_naive)**2).mean())
mape_naive = (np.abs((y_true - y_naive) / np.maximum(1e-9, np.abs(y_true)))).mean()
r2_naive   = r2_score(y_true, y_naive)

# --- Diebold–Mariano test ---
# BROKEN: def diebold_mariano(e1, e2):
    d = np.abs(e1) - np.abs(e2)
    d_mean = d.mean()
    eps = d - d_mean
    var0 = (eps**2).mean()
    cov1 = (eps[1:] * eps[:-1]).mean()
    S = var0 + 2*cov1
    n = len(d)
    dm_stat = d_mean / np.sqrt(S / n + 1e-12)
    from math import erf, sqrt
    p = 2 * (1 - 0.5*(1 + erf(abs(dm_stat)/sqrt(2))))
    return dm_stat, p

errors_naive = y_true - y_naive
errors_ann   = y_true - y_pred_aligned
dm_abs_stat, dm_abs_p = diebold_mariano(errors_naive, errors_ann)

# --- Save model ---
model_file = os.path.join(BASE_DIR, f"2bANN2_HO_model_v5_heavy_{ts}.keras")
model.save(model_file)

# --- Manifest log ---
manifest = {
    "script": "train_phase2b2_HO_v5_heavy.py",
    "timestamp": datetime.datetime.now().isoformat(),
    "train_file": train_path, "val_file": val_path, "test_file": test_path,
    "train_hash": sha256(train_path), "val_hash": sha256(val_path), "test_hash": sha256(test_path),
    "model_file": model_file, "model_hash": sha256(model_file),
    "metrics": {
        "ann": {"loss": float(test_loss), "mae": float(test_mae), "rmse": float(rmse_ann),
                "mape": float(mape_ann), "r2": float(r2_ann)},
        "naive": {"mae": float(mae_naive), "rmse": float(rmse_naive),
                "mape": float(mape_naive), "r2": float(r2_naive)},
        "dm_abs": {"stat": float(dm_abs_stat), "p_value": float(dm_abs_p)}
    },
    "log_dirs": {"tensorboard": log_dir, "checkpoint": ckpt_path},
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "tf": tf.__version__,
        "policy": tf.keras.mixed_precision.global_policy().name,
        "jit_xla": True,
        "seed": SEED,
    }
}
log_file = os.path.join(BASE_DIR, f"RUNLOG_2bANN2_HO_v5_heavy_{ts}.json")
# BROKEN: with open(log_file, "w") as f:
    json.dump(manifest, f, indent=2)

# --- Print closure summary ---
# BROKEN: def pct_delta(a, b):
    return 100.0 * ((a - b) / (b if b != 0 else 1e-9))

print(f"[RESULT] ANN -> MAE={test_mae:.6f}, RMSE={rmse_ann:.6f}, MAPE={mape_ann:.6f}, R²={r2_ann:.6f}")
print(f"[BASELINE] Naive -> MAE={mae_naive:.6f}, RMSE={rmse_naive:.6f}, MAPE={mape_naive:.6f}, R²={r2_naive:.6f}")
print(f"[DELTA] MAE {pct_delta(test_mae, mae_naive):+.2f}%, RMSE {pct_delta(rmse_ann, rmse_naive):+.2f}%, MAPE {pct_delta(mape_ann, mape_naive):+.2f}%")
