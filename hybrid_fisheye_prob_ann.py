# hybrid_fisheye_prob_ann.py
# Purpose: hybrid fisheye/prob ann with pooled encoder fallback, deterministic CSV loader,
#          SHA256 manifesting, mu pretrain then conservative probabilistic fine-tune,
#          residual-target learning (target - last close), per-feature normalization,
#          stabilized/bounded log-var handling, and conservative Phase-2 settings.
# Save as: hybrid_fisheye_prob_ann.py
# Run (PowerShell):
#   cd "C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080"
#   python hybrid_fisheye_prob_ann.py

# ---------------------------
# Explicit CSV paths (edit only if you must)
TRAIN_CSV = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_training.csv"
VAL_CSV   = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_validation.csv"
TEST_CSV  = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_testing.csv"

# ---------------------------
# Preflight check: files exist and SHA256; fail loudly if missing
import os, sys, hashlib, time, json
# BROKEN: def file_sha256(path, chunk_size=1 << 20):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         while True:
            b = f.read(chunk_size)
# BROKEN:             if not b:
                break
            h.update(b)
    return h.hexdigest()

_paths = [TRAIN_CSV, VAL_CSV, TEST_CSV]
_missing = [p for p in _paths if not os.path.isfile(p)]
# BROKEN: if _missing:
    raise FileNotFoundError("Missing CSV(s). Edit TRAIN_CSV/VAL_CSV/TEST_CSV or place files at these paths: " + ", ".join(_missing))
# BROKEN: for _p in _paths:
    print("CSV found:", os.path.abspath(_p), "| sha256:", file_sha256(_p))
del _paths, _missing, _p

# ---------------------------
# Imports and config
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from scipy import stats

# ---------------------------
# Run identity and hyperparameters
RUN_NAME = "hybrid_fisheye_prob_ann"
SEED = 77
CONTEXT_LEN = 128
FEATURES_PER_T = 4
EMBED_DIM = 256
ATTN_DIM = 128
ATTN_HEADS = 4
DROPOUT = 0.10
L2W = 1e-6
EPOCHS_P1 = 16
EPOCHS_P2 = 4
BATCH = 64
LR = 1e-4
LR_P2 = 1e-5
DATE_FMT = "%d-%b-%y"
MIN_LOG_VAR = 1e-6

# ---------------------------
# Determinism
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Deterministic CSV loader with header normalization
# BROKEN: def load_df(path, date_fmt=DATE_FMT):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.capitalize()
    required = ["Date", "Open", "High", "Low", "Close"]
    missing_cols = [c for c in required if c not in df.columns]
# BROKEN:     if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
# BROKEN:     try:
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format=date_fmt)
# BROKEN:     except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise", infer_datetime_format=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ---------------------------
# Load datasets
df_train = load_df(TRAIN_CSV, date_fmt=DATE_FMT)
df_val   = load_df(VAL_CSV,   date_fmt=DATE_FMT)
df_test  = load_df(TEST_CSV,  date_fmt=DATE_FMT)

date_checks_head = [
    {"split": "train", "dates_head": [str(d) for d in df_train["Date"].head(12).tolist()]},
    {"split": "val",   "dates_head": [str(d) for d in df_val["Date"].head(12).tolist()]},
    {"split": "test",  "dates_head": [str(d) for d in df_test["Date"].head(12).tolist()]},
]

# ---------------------------
# Sequence builder (context -> next-step Close)
# BROKEN: def build_sequences(df, context_len=CONTEXT_LEN):
    ohlc = df[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float32)
    dates = df["Date"].to_numpy()
    X, y, meta = [], [], []
# BROKEN:     for i in range(context_len, len(ohlc) - 1):
        ctx = ohlc[i - context_len:i]                       # context window (context_len x 4)
        target = ohlc[i, 3]                                 # next-step Close (row i)
        X.append(ctx)
        y.append(target)
        meta.append({
            "window_end_idx": i - 1,
            "window_end_date": str(pd.Timestamp(dates[i - 1])),
            "target_idx": i,
            "target_date": str(pd.Timestamp(dates[i])),
            "window_last_close": float(ohlc[i - 1, 3]),
            "target_close": float(ohlc[i, 3]),
            "persistence_close": float(ohlc[i - 1, 3])
        })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta

X_train, y_train, meta_train = build_sequences(df_train)
X_val,   y_val,   meta_val   = build_sequences(df_val)
X_test,  y_test,  meta_test  = build_sequences(df_test)

# ---------------------------
# Convert targets to residuals (target - last close)
# BROKEN: def _to_residuals(y_arr, meta):
    deltas = []
# BROKEN:     for i, v in enumerate(y_arr):
        last_close = float(meta[i]["window_last_close"])
        deltas.append(float(v) - last_close)
    return np.asarray(deltas, dtype=np.float32)

y_train = _to_residuals(y_train, meta_train)
y_val   = _to_residuals(y_val,   meta_val)
y_test  = _to_residuals(y_test,  meta_test)

# ---------------------------
# Per-feature normalization (fit on train)
train_flat = X_train.reshape(-1, X_train.shape[-1])
feat_mean = train_flat.mean(axis=0)
feat_std = train_flat.std(axis=0) + 1e-9
X_train = (X_train - feat_mean) / feat_std
X_val   = (X_val   - feat_mean) / feat_std
X_test  = (X_test  - feat_mean) / feat_std

# Standardize residual targets (fit on train residuals)
y_mean = float(np.mean(y_train))
y_std = float(np.std(y_train)) + 1e-9
y_train = (y_train - y_mean) / y_std
y_val   = (y_val   - y_mean) / y_std
y_test  = (y_test   - y_mean) / y_std

# After standardization, zero-centered targets -> init bias to 0
train_mean = 0.0

# ---------------------------
# Model components
# BROKEN: def pooled_encoder(x, embed_dim=EMBED_DIM, dropout=DROPOUT, l2=L2W):
    x = layers.TimeDistributed(layers.Dense(embed_dim, activation="relu",
                                            kernel_regularizer=regularizers.l2(l2)))(x)
    x = layers.TimeDistributed(layers.Dense(embed_dim, activation="relu",
                                            kernel_regularizer=regularizers.l2(l2)))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    return x

# BROKEN: def probabilistic_head(x, l2=L2W, init_bias=None, logvar_init=-3.0):
    bias_init = initializers.Constant(init_bias) if init_bias is not None else "zeros"
    mu = layers.Dense(1, activation="linear", kernel_regularizer=regularizers.l2(l2),
                    name="mu", bias_initializer=bias_init)(x)
    log_var_raw = layers.Dense(1, activation="linear", kernel_regularizer=regularizers.l2(l2),
                            name="log_var_raw", bias_initializer=initializers.Constant(logvar_init))(x)
    return mu, log_var_raw

# Bounded log-var for numerical stability (tf)
# BROKEN: def stabilized_log_var_tf(log_var_raw):
    return tf.clip_by_value(log_var_raw, -8.0, 6.0)

# Gaussian NLL operates on standardized targets (unit variance) and is stable due to clipping
# BROKEN: def gaussian_nll(y_true, mu, log_var_raw, eps=1e-6):
    log_var = stabilized_log_var_tf(log_var_raw)
    var = tf.exp(log_var) + eps
    return 0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var)

# BROKEN: def build_model(context_len=CONTEXT_LEN, feat_dim=FEATURES_PER_T, init_bias=None):
    inputs = layers.Input(shape=(context_len, feat_dim), name="context")
    x = pooled_encoder(inputs)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2W))(x)
    x = layers.Dropout(DROPOUT)(x)
    mu, log_var_raw = probabilistic_head(x, l2=L2W, init_bias=init_bias)
    model = models.Model(inputs=inputs, outputs=[mu, log_var_raw], name=RUN_NAME)
    return model

# ---------------------------
# Build and Phase 1 pretrain (MAE on mu) with EarlyStopping
model = build_model(init_bias=train_mean)
model_p1 = models.Model(inputs=model.inputs, outputs=model.get_layer("mu").output, name=f"{RUN_NAME}_p1")
opt_p1 = tf.keras.optimizers.Adam(LR)
model_p1.compile(optimizer=opt_p1, loss="mae")
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
hist_p1 = model_p1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_P1,
    batch_size=BATCH,
    callbacks=[es],
    verbose=1
)

# Transfer weights where layer names match
# BROKEN: for layer in model.layers:
# BROKEN:     try:
        src = model_p1.get_layer(layer.name)
        layer.set_weights(src.get_weights())
# BROKEN:     except Exception:
        pass

# ---------------------------
# MU-ONLY EVALUATION (after weight transfer)
mu_val_p1 = model_p1.predict(X_val, verbose=0).reshape(-1)
mu_test_p1 = model_p1.predict(X_test, verbose=0).reshape(-1)

# Convert mu preds back to absolute (denormalize residuals then add persistence)
persistence_test = np.array([m["persistence_close"] for m in meta_test], dtype=np.float32)
mu_test_p1_denorm = (mu_test_p1 * y_std) + y_mean
mu_test_p1_abs = persistence_test + mu_test_p1_denorm
targets_test_abs = np.array([m["target_close"] for m in meta_test], dtype=np.float32)

mae_model_p1 = float(np.mean(np.abs(targets_test_abs - mu_test_p1_abs)))
mae_persistence = float(np.mean(np.abs(targets_test_abs - persistence_test)))
t_stat_p1, p_val_p1 = stats.ttest_rel(np.abs(targets_test_abs - persistence_test), np.abs(targets_test_abs - mu_test_p1_abs))

manifest_interim = {
    "run_name": RUN_NAME,
    "stage": "mu_only_pretrain",
    "train_mean": float(train_mean),
    "mae_model_p1": mae_model_p1,
    "mae_persistence": mae_persistence,
    "paired_ttest_t": float(t_stat_p1),
    "paired_ttest_p": float(p_val_p1)
}
# BROKEN: with open(f"{RUN_NAME}_mu_only_interim.json", "w", encoding="utf-8") as _f:
    json.dump(manifest_interim, _f, indent=2)
print({"stage":"mu_only_pretrain","mae_model_p1":mae_model_p1,"mae_persistence":mae_persistence,"paired_ttest_p":float(p_val_p1)})

# ---------------------------
# Phase 2 fine-tune (probabilistic NLL) with conservative LR, gradient clipping, and EarlyStopping
# BROKEN: if EPOCHS_P2 > 0:
    optimizer = tf.keras.optimizers.Adam(LR_P2, clipnorm=0.25)

    @tf.function
# BROKEN:     def train_step(xb, yb):
        ...
# BROKEN:         with tf.GradientTape() as tape:
            mu_pred, log_var_raw_pred = model(xb, training=True)
            nll = tf.reduce_mean(gaussian_nll(yb, mu_pred, log_var_raw_pred))
            reg_losses = tf.add_n(model.losses) if model.losses else 0.0
            loss = nll + reg_losses
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, nll

    @tf.function
# BROKEN:     def val_step(xb, yb):
        mu_pred, log_var_raw_pred = model(xb, training=False)
        nll = tf.reduce_mean(gaussian_nll(yb, mu_pred, log_var_raw_pred))
        reg_losses = tf.add_n(model.losses) if model.losses else 0.0
        loss = nll + reg_losses
        return loss, nll

# BROKEN:     def run_epoch(X, y, step_fn, batch=BATCH):
        n = X.shape[0]
        idx = tf.range(n)
        idx = tf.random.shuffle(idx)
        losses = []
# BROKEN:         for i in range(0, n, batch):
            sel = idx[i:i+batch]
            l, nll = step_fn(tf.gather(X, sel), tf.gather(y, sel))
            losses.append(l)
        return float(tf.reduce_mean(losses))

    es_p2 = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    train_losses_p2, val_losses_p2 = [], []
# BROKEN:     for e in range(EPOCHS_P2):
        tl = run_epoch(X_train, y_train, train_step, batch=BATCH)
        vl, _ = val_step(X_val, y_val)
        train_losses_p2.append(tl)
        val_losses_p2.append(float(vl))
        print(f"Prob epoch {e+1}/{EPOCHS_P2} - loss: {tl:.6f} - val_loss: {vl:.6f}")
        # EarlyStopping check using simple logic
# BROKEN:         if e >= 2 and val_losses_p2[-1] > min(val_losses_p2[:-2]):
            break

# ---------------------------
# Predict on val and test, diagnostics
mu_val_raw, log_var_val_raw = model.predict(X_val, verbose=0)
mu_test_raw, log_var_test_raw = model.predict(X_test, verbose=0)

mu_val = np.asarray(mu_val_raw).reshape(-1)
mu_test = np.asarray(mu_test_raw).reshape(-1)
log_var_val_raw = np.asarray(log_var_val_raw).reshape(-1)
log_var_test_raw = np.asarray(log_var_test_raw).reshape(-1)

# stable softplus/logvar in numpy for reporting (avoid overflow)
# BROKEN: def _stabilize_logvar_np(raw):
    raw = np.asarray(raw, dtype=np.float64)
    out = np.clip(raw, -8.0, 6.0)
    return out

log_var_test_stable = _stabilize_logvar_np(log_var_test_raw)

# reconstruct absolute predictions from residual-model outputs (denormalize residuals then add persistence)
persistence = np.array([m["persistence_close"] for m in meta_test], dtype=np.float32)
mu_test_denorm = (mu_test * y_std) + y_mean
model_abs_pred = persistence + mu_test_denorm
targets = np.array([m["target_close"] for m in meta_test], dtype=np.float32)

# diagnostic head (first 16)
diagnostic_head = []
# BROKEN: for i in range(min(16, len(meta_test))):
    diagnostic_head.append({
        "window_end_idx": meta_test[i]["window_end_idx"],
        "window_end_date": meta_test[i]["window_end_date"],
        "target_idx": meta_test[i]["target_idx"],
        "target_date": meta_test[i]["target_date"],
        "window_last_close": meta_test[i]["window_last_close"],
        "target_close": meta_test[i]["target_close"],
        "persistence_close": meta_test[i]["persistence_close"],
        "mu_residual_std": float(mu_test[i]),
        "mu_residual_denorm": float(mu_test_denorm[i]),
        "mu_abs": float(model_abs_pred[i]),
        "log_var_raw": float(log_var_test_raw[i]),
        "log_var_report": float(log_var_test_stable[i])
    })

# ---------------------------
# Persistence baseline comparison MAE + paired t-test (on absolute predictions)
mae_persistence = float(np.mean(np.abs(targets - persistence)))
mae_model = float(np.mean(np.abs(targets - model_abs_pred)))
mae_improvement = float(mae_persistence - mae_model)
rel_improvement = float(mae_improvement / mae_persistence) if mae_persistence != 0 else float("nan")

err_p = np.abs(targets - persistence)
err_m = np.abs(targets - model_abs_pred)
t_stat, p_val = stats.ttest_rel(err_p, err_m)

persistence_comparison = {
    "persistence_mae": mae_persistence,
    "model_mae": mae_model,
    "absolute_improvement": mae_improvement,
    "relative_improvement": rel_improvement,
    "paired_ttest_t": float(t_stat),
    "paired_ttest_p": float(p_val)
}

# ---------------------------
# Final metrics, manifest and provenance
final_train_loss = float(train_losses_p2[-1]) if (EPOCHS_P2 > 0 and 'train_losses_p2' in locals() and len(train_losses_p2)>0) else float(hist_p1.history["loss"][-1])
final_val_loss = float(val_losses_p2[-1]) if (EPOCHS_P2 > 0 and 'val_losses_p2' in locals() and len(val_losses_p2)>0) else float(hist_p1.history["val_loss"][-1])
# compute test NLL on standardized residuals using stable log_var
residuals_std = (((targets - persistence) - y_mean) / (y_std + 1e-12)) - mu_test
term = np.log(np.exp(log_var_test_stable) + 1e-6) + (residuals_std ** 2) / (np.exp(log_var_test_stable) + 1e-6)
nll_test = float(np.mean(0.5 * term))

manifest = {
    "run_name": RUN_NAME,
    "params": {
        "date_fmt": DATE_FMT,
        "context_len": CONTEXT_LEN,
        "features_per_t": FEATURES_PER_T,
        "embed_dim": EMBED_DIM,
        "attn_dim": ATTN_DIM,
        "attn_heads": ATTN_HEADS,
        "dropout": DROPOUT,
        "l2_weight": L2W,
        "seed": SEED,
        "epochs_p1": EPOCHS_P1,
        "epochs_p2": EPOCHS_P2,
        "batch": BATCH,
        "lr_pretrain": LR,
        "lr_p2": LR_P2
    },
    "files": {
        "train_path": TRAIN_CSV,
        "val_path": VAL_CSV,
        "test_path": TEST_CSV,
        "train_sha256": file_sha256(TRAIN_CSV),
        "val_sha256": file_sha256(VAL_CSV),
        "test_sha256": file_sha256(TEST_CSV),
        "df_train_sha256": hashlib.sha256(df_train.to_numpy().tobytes()).hexdigest(),
        "df_val_sha256": hashlib.sha256(df_val.to_numpy().tobytes()).hexdigest(),
        "df_test_sha256": hashlib.sha256(df_test.to_numpy().tobytes()).hexdigest()
    },
    "metrics": {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "test_nll": nll_test,
        "persistence_comparison": persistence_comparison
    },
    "debug_checks": {
        "date_checks_head": date_checks_head,
        "diagnostic_head": diagnostic_head,
        "train_mean": float(train_mean),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std)
    },
    "provenance": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "script": "hybrid_fisheye_prob_ann.py",
        "python_version": f"{sys.version.split()[0]}",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tensorflow": tf.__version__
    }
}

manifest_filename = f"{RUN_NAME}_manifest.json"
# BROKEN: with open(manifest_filename, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

summary = {
    "run": RUN_NAME,
    "epochs_p1": EPOCHS_P1,
    "epochs_p2": EPOCHS_P2,
    "context_len": CONTEXT_LEN,
    "final_train_loss": manifest["metrics"]["final_train_loss"],
    "final_val_loss": manifest["metrics"]["final_val_loss"],
    "test_nll": manifest["metrics"]["test_nll"],
    "persistence_model_mae": manifest["metrics"]["persistence_comparison"]["model_mae"],
    "persistence_baseline_mae": manifest["metrics"]["persistence_comparison"]["persistence_mae"],
    "paired_ttest_p": manifest["metrics"]["persistence_comparison"]["paired_ttest_p"],
    "manifest": manifest_filename
}

print(summary)
