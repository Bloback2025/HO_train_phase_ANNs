# HO_run_delta.py
# Date: 2025-10-24
# Purpose:
#   Train and evaluate a deterministic GRU forecaster on Heating Oil OHLC data.
#   - Normalizes headers
#   - Verifies chronological separation (prints first/last dates + head/tail)
#   - Scales features using train stats (applied to val/test)
#   - Runs a deterministic unit test for target indexing
#   - Uses a GRU-based model with regularization and stabilized training
#   - Writes an audit-safe manifest with hashes, counts, dates, and metrics

import os
import time
import json
import hashlib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow import keras

# --- Config (explicit paths) ---
TRAIN_CSV = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_training.csv"
VAL_CSV   = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_validation.csv"
TEST_CSV  = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_testing.csv"

EPOCHS       = 50
BATCH        = 128
LEARNING_RATE= 1e-4
SEED         = 5080
CONTEXT_CAP  = 256
CONTEXT_MODEL_MAX = 64  # conservative window for GRU

# --- Determinism ---
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# --- Utilities ---
# BROKEN: def sha256_file(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# BROKEN: def load_df_with_normalized_columns(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# --- Load dataframes and print chronology for audit ---
df_train = load_df_with_normalized_columns(TRAIN_CSV)
df_val   = load_df_with_normalized_columns(VAL_CSV)
df_test  = load_df_with_normalized_columns(TEST_CSV)

print("TRAIN date range:", df_train['Date'].min(), df_train['Date'].max())
print("VAL   date range:", df_val['Date'].min(), df_val['Date'].max())
print("TEST  date range:", df_test['Date'].min(), df_test['Date'].max())

print("TRAIN head:", df_train.head(3).to_dict(orient='list'))
print("TRAIN tail:", df_train.tail(3).to_dict(orient='list'))
print("VAL head:", df_val.head(3).to_dict(orient='list'))
print("VAL tail:", df_val.tail(3).to_dict(orient='list'))
print("TEST head:", df_test.head(3).to_dict(orient='list'))
print("TEST tail:", df_test.tail(3).to_dict(orient='list'))

# --- Convert to numeric arrays (OHLC) ---
# BROKEN: def df_to_ohlc_array(df):
    arr = df[["Open", "High", "Low", "Close"]].values.astype(np.float32)
    return arr

train = df_to_ohlc_array(df_train)
val   = df_to_ohlc_array(df_val)
test  = df_to_ohlc_array(df_test)

print("Train shape:", train.shape)
print("Val shape:", val.shape)
print("Test shape:", test.shape)

# --- Derive safe context length based on smallest split, capped reasonably ---
min_rows = min(train.shape[0], val.shape[0], test.shape[0])
# BROKEN: if min_rows < 5:
    raise ValueError(f"Too few rows in splits; smallest has {min_rows} rows.")
derived_context = int(min(CONTEXT_CAP, max(4, min_rows - 1)))
CONTEXT_LEN = min(derived_context, CONTEXT_MODEL_MAX)
print(f"Derived CONTEXT_LEN={CONTEXT_LEN} (derived={derived_context}, model_cap={CONTEXT_MODEL_MAX}, smallest_split={min_rows})")

# --- Scaling using train stats (apply same transform to val/test) ---
train_mean = train.mean(axis=0, keepdims=True)
train_std  = train.std(axis=0, keepdims=True) + 1e-8

# BROKEN: def scale(arr):
    return (arr - train_mean) / train_std

# BROKEN: def unscale_close(v_scaled):
    # v_scaled shape (N,1) in scaled space; return original-scale (N,1)
    mean_close = train_mean[0, 3]
    std_close = train_std[0, 3]
    return v_scaled * std_close + mean_close

train_s = scale(train)
val_s   = scale(val)
test_s  = scale(test)

# --- Sequence builder (strict stack) ---
# BROKEN: def make_sequences(arr, context_len):
    N = len(arr) - context_len
# BROKEN:     if N <= 0:
        raise ValueError(f"Not enough rows ({len(arr)}) for context_len={context_len}")
    X = [arr[i:i+context_len] for i in range(N)]
    y = [arr[i+context_len, 3] for i in range(N)]  # next Close
    X = np.stack(X, axis=0).astype(np.float32)     # (N, context_len, 4)
    y = np.asarray(y, dtype=np.float32).reshape(-1,1)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    return X, y

# --- Unit test for target indexing (synthetic) ---
tmp = np.arange((CONTEXT_LEN + 3) * 4).reshape(-1,4).astype(np.float32)
Xtmp, ytmp = make_sequences(tmp, CONTEXT_LEN)
# BROKEN: if not np.allclose(ytmp[0,0], tmp[CONTEXT_LEN,3]):
    raise AssertionError("Target indexing unit test failed: y_next != arr[context_len,3]")

# --- Build sequences for real data ---
X_train, y_train = make_sequences(train_s, CONTEXT_LEN)
X_val,   y_val   = make_sequences(val_s,   CONTEXT_LEN)
X_test,  y_test  = make_sequences(test_s,  CONTEXT_LEN)

# --- Defensive checks ---
# BROKEN: def ensure_ok(X, y, name):
# BROKEN:     if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"{name}: X/y must be numpy.ndarray; got X={type(X)} y={type(y)}")
# BROKEN:     if X.ndim != 3 or X.shape[1:] != (CONTEXT_LEN, 4):
        raise ValueError(f"{name}: X shape must be (N,{CONTEXT_LEN},4); got {X.shape}")
# BROKEN:     if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError(f"{name}: y shape must be (N,1); got {y.shape}")
# BROKEN:     if X.dtype != np.float32 or y.dtype != np.float32:
        raise TypeError(f"{name}: dtypes must be float32; got X={X.dtype}, y={y.dtype}")

ensure_ok(X_train, y_train, "TRAIN")
ensure_ok(X_val,   y_val,   "VAL")
ensure_ok(X_test,  y_test,  "TEST")

print("X_train dtype/shape:", X_train.dtype, X_train.shape)
print("y_train dtype/shape:", y_train.dtype, y_train.shape)
print("X_val dtype/shape:", X_val.dtype, X_val.shape)
print("y_val dtype/shape:", y_val.dtype, y_val.shape)

# --- GRU forecaster (regularized, stable) ---
inputs = keras.Input(shape=(CONTEXT_LEN, 4))
x = keras.layers.GRU(64, kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
outputs = keras.layers.Dense(1)(x)
forecaster = keras.Model(inputs, outputs)

forecaster.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

# --- Callbacks ---
es = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
rlr= callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
ck = callbacks.ModelCheckpoint("ho2bann_gru_best.keras", monitor="val_loss", save_best_only=True)
csv= callbacks.CSVLogger("ho2bann_gru_log.csv", append=False)

# --- Fit ---
hist = forecaster.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[es, rlr, ck, csv],
    verbose=2
)

# --- Evaluate on test set (scaled) ---
test_metrics_scaled = forecaster.evaluate(X_test, y_test, verbose=2)
pred_scaled = forecaster.predict(X_test, verbose=0)
mae_model_scaled = float(np.mean(np.abs(pred_scaled - y_test)))
persist_pred_scaled = X_test[:, -1, 3].reshape(-1,1)
mae_persist_scaled = float(np.mean(np.abs(persist_pred_scaled - y_test)))

# --- Convert to original scale for reporting ---
y_test_orig = unscale_close(y_test)
pred_orig = unscale_close(pred_scaled)
persist_orig = unscale_close(persist_pred_scaled)
mae_model_orig = float(np.mean(np.abs(pred_orig - y_test_orig)))
mae_persist_orig = float(np.mean(np.abs(persist_orig - y_test_orig)))

print({"mae_model_scaled": mae_model_scaled, "mae_persistence_scaled": mae_persist_scaled, "keras_eval_scaled": list(map(float, test_metrics_scaled))})
print({"mae_model_orig": mae_model_orig, "mae_persistence_orig": mae_persist_orig})

# --- Manifest (audit-safe) ---
manifest = {
    "run_id": int(time.time()),
    "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
    "script": "HO_run_delta.py",
    "seed": SEED,
    "context_len": CONTEXT_LEN,
    "paths": {"train_csv": TRAIN_CSV, "val_csv": VAL_CSV, "test_csv": TEST_CSV},
    "date_ranges": {
        "train_min": str(df_train['Date'].min()), "train_max": str(df_train['Date'].max()),
        "val_min": str(df_val['Date'].min()),     "val_max": str(df_val['Date'].max()),
        "test_min": str(df_test['Date'].min()),   "test_max": str(df_test['Date'].max())
    },
    "counts": {"train_rows": int(train.shape[0]), "val_rows": int(val.shape[0]), "test_rows": int(test.shape[0])},
    "hashes": {"train_sha256": sha256_file(TRAIN_CSV), "val_sha256": sha256_file(VAL_CSV), "test_sha256": sha256_file(TEST_CSV)},
    "metrics_scaled": {"mae_model_scaled": mae_model_scaled, "mae_persistence_scaled": mae_persist_scaled, "keras_eval_scaled": list(map(float, test_metrics_scaled))},
    "metrics_original": {"mae_model_orig": mae_model_orig, "mae_persistence_orig": mae_persist_orig},
    "closure": "sealed"
}

# BROKEN: with open("ho2bann_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Manifest written: ho2bann_manifest.json")
