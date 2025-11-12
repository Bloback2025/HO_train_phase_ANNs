# Name: hybrid_fisheye_prob_ann_patch_run.py
# Date: 2025-10-24
# Purpose: Patch of hybrid_fisheye_prob_ann for immediate diagnostics and remediation:
#  - auxiliary MAE on mu to prevent sigma inflation
#  - clipped/bounded log_sigma output
#  - pretrain toggle and reduced-capacity mode
#  - diagnostic dumps (mu/sigma/y/pers scaled + unscaled) and sequence-date alignment
#  - manifest, checkpoints, CSV logs for audit

import os, json, hashlib, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ===== Reproducibility =====
SEED = 77
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ===== Paths =====
TRAIN_CSV = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_training.csv"
VAL_CSV   = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_validation.csv"
TEST_CSV  = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_testing.csv"

# ===== Run flags =====
RUN_NAME            = "RunPatch_auxMAE_clipSigma_ctx128"  # set name per run
PRETRAIN_ENABLED    = True
CONTEXT_LEN_FIXED   = 128
LEARN_RATE_PRE      = 1e-3
LEARN_RATE_FT       = 1e-4
DROPOUT             = 0.30
L2W                 = 3e-4
BATCH_PRE           = 64
BATCH_FT            = 128
EPOCHS_PRE          = 30
EPOCHS_FT           = 60
AUX_MAE_ALPHA       = 0.1    # weight on MAE(mu) auxiliary loss
LOGSIGMA_CLIP_MIN   = -5.0   # lower bound for log_sigma
LOGSIGMA_CLIP_MAX   = 2.0    # upper bound for log_sigma
REDUCED_CAPACITY    = False  # if True, shrink MLP_WIDTHS for quick tests

# ===== Model dims =====
FEATURES_PER_T = 4   # O,H,L,C
EMBED_DIM      = 64
ATTN_DIM       = 128
ATTN_HEADS     = 4
MLP_WIDTHS     = [256, 256, 128] if not REDUCED_CAPACITY else [128, 128]
MASK_RATIO     = 0.15

# ===== Utilities =====
# BROKEN: def sha256_file(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

# BROKEN: def write_manifest(meta: dict, path: str):
# BROKEN:     with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# BROKEN: def load_df(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.sort_values("Date").reset_index(drop=True)
    assert all(c in df.columns for c in ["Open","High","Low","Close","Date"]), "Missing required columns."
    return df

# BROKEN: def df_to_arr(df):
    return df[["Open","High","Low","Close"]].values.astype(np.float32)

# BROKEN: def scale_with_train_stats(train_arr, val_arr, test_arr):
    mean = train_arr.mean(axis=0, keepdims=True)
    std  = train_arr.std(axis=0, keepdims=True) + 1e-8
    def s(a): return (a - mean) / std
    return s(train_arr), s(val_arr), s(test_arr), mean, std

# BROKEN: def make_sequences(arr, context_len):
    N = len(arr) - context_len
# BROKEN:     if N <= 0:
        raise ValueError(f"Not enough rows ({len(arr)}) for context_len={context_len}")
    X = np.stack([arr[i:i+context_len] for i in range(N)], axis=0).astype(np.float32)
    y = np.asarray([arr[i+context_len, 3] for i in range(N)], dtype=np.float32).reshape(-1,1)
    return np.ascontiguousarray(X), np.ascontiguousarray(y)

def mae(a, b): return float(np.mean(np.abs(a - b)))

# BROKEN: def dm_test(e_model, e_pers):
    d = e_model - e_pers
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = d_bar / np.sqrt(var_d / len(d))
    return float(dm_stat)

# ===== Fish-eye encoder (same as before) =====
# BROKEN: def fisheye_encoder(seq_in):
    x = layers.Dense(EMBED_DIM, activation=None,
                    kernel_regularizer=regularizers.l2(L2W))(seq_in)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    local = layers.Conv1D(EMBED_DIM, kernel_size=5, padding="same",
                        activation="relu", kernel_regularizer=regularizers.l2(L2W))(x)
    global_ = layers.Conv1D(EMBED_DIM, kernel_size=31, padding="same",
                            activation="relu", kernel_regularizer=regularizers.l2(L2W))(x)
    x = layers.Concatenate()([local, global_])
    attn = layers.MultiHeadAttention(num_heads=ATTN_HEADS, key_dim=ATTN_DIM, dropout=DROPOUT)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(2*EMBED_DIM, activation="relu", kernel_regularizer=regularizers.l2(L2W))(x)
    ff = layers.Dropout(DROPOUT)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# ===== Self-supervised masked autoencoder =====
# BROKEN: def build_masked_autoencoder(context_len):
    seq_in = layers.Input(shape=(context_len, FEATURES_PER_T), name="seq_in")
    encoded = fisheye_encoder(seq_in)
    recon = layers.Dense(FEATURES_PER_T, activation=None, name="recon")(encoded)
    return models.Model(seq_in, recon, name="masked_autoencoder")

# BROKEN: def pretrain_autoencoder(mae_model, seqs, epochs=EPOCHS_PRE):
    N, T, F = seqs.shape
    mask  = (np.random.rand(N, T, 1) < MASK_RATIO).astype(np.float32)
    noise = np.random.normal(0, 0.1, size=seqs.shape).astype(np.float32)
    masked = seqs * (1 - mask) + noise * mask
    mae_model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE_PRE), loss="mse")
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    ck  = callbacks.ModelCheckpoint(f"{RUN_NAME}_mae_best.keras", monitor="val_loss", save_best_only=True)
    csv = callbacks.CSVLogger(f"{RUN_NAME}_mae_log.csv", append=False)
    hist = mae_model.fit(masked, seqs, batch_size=BATCH_PRE, epochs=epochs,
                        validation_split=0.1, callbacks=[es, rlr, ck, csv], verbose=2)
    return hist

# ===== Probabilistic forecasting head with clipped log_sigma and auxiliary MAE =====
# BROKEN: def nll_gaussian(y_true, y_pred):
    mu = y_pred[:, 0:1]
    log_sigma = y_pred[:, 1:2]
    sigma = tf.exp(log_sigma)
    nll = 0.5 * tf.math.log(2.0 * np.pi) + log_sigma + 0.5 * tf.square((y_true - mu) / sigma)
    return tf.reduce_mean(nll)

# BROKEN: def mae_mu_loss(y_true, y_pred):
    mu = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(y_true - mu))

# BROKEN: def build_prob_forecaster_clipped(context_len):
    seq_in = layers.Input(shape=(context_len, FEATURES_PER_T), name="seq_in")
    encoded = fisheye_encoder(seq_in)
    global_pool = layers.GlobalAveragePooling1D()(encoded)
    last_step   = layers.Lambda(lambda z: z[:, -1, :])(encoded)
    z = layers.Concatenate()([global_pool, last_step])
# BROKEN:     for w in MLP_WIDTHS:
        z = layers.Dense(w, activation="relu", kernel_regularizer=regularizers.l2(L2W))(z)
        z = layers.Dropout(DROPOUT)(z)
    mu = layers.Dense(1, activation="linear", name="mu")(z)
    # raw log_sigma then clip via Lambda
    log_sigma_raw = layers.Dense(1, activation="linear", name="log_sigma_raw")(z)
    log_sigma_clipped = layers.Lambda(lambda t: tf.clip_by_value(t, LOGSIGMA_CLIP_MIN, LOGSIGMA_CLIP_MAX),
                                    name="log_sigma")(log_sigma_raw)
    out = layers.Concatenate(name="mu_logsigma")([mu, log_sigma_clipped])
    return models.Model(seq_in, out, name="hybrid_prob_forecaster_clipped")

# BROKEN: def finetune_prob_forecaster_aux(model, X_seq, y_next, epochs=EPOCHS_FT):
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    ck  = callbacks.ModelCheckpoint(f"{RUN_NAME}_forecaster_best.keras", monitor="val_loss", save_best_only=True)
    csv = callbacks.CSVLogger(f"{RUN_NAME}_forecaster_log.csv", append=False)

    # combined loss uses NLL + alpha * MAE(mu)
    alpha = AUX_MAE_ALPHA
# BROKEN:     def combined_loss(y_true, y_pred):
        return nll_gaussian(y_true, y_pred) + alpha * mae_mu_loss(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE_FT), loss=combined_loss)
    hist = model.fit(X_seq, y_next, batch_size=BATCH_FT, epochs=epochs,
                    validation_split=0.2, callbacks=[es, rlr, ck, csv], verbose=2)
    return hist

# ===== Calibration coverage =====
# BROKEN: def calibration_coverage(mu, sigma, y, bands=(1.0, 2.0)):
    cov = {}
# BROKEN:     for k in bands:
        inside = np.abs(y - mu) <= (k * sigma)
        cov[f"coverage_{k}sigma"] = float(np.mean(inside))
    cov["sharpness_mean_sigma"] = float(np.mean(sigma))
    return cov

# BROKEN: def persistence_pred_scaled(X_seq):
    return X_seq[:, -1, 3].reshape(-1,1)

# ===== Main =====
# BROKEN: if __name__ == "__main__":
    # Load dataframes
    df_train = load_df(TRAIN_CSV)
    df_val   = load_df(VAL_CSV)
    df_test  = load_df(TEST_CSV)

    print("TRAIN:", df_train["Date"].min(), "->", df_train["Date"].max(), "rows:", len(df_train))
    print("VAL  :", df_val["Date"].min(),   "->", df_val["Date"].max(),   "rows:", len(df_val))
    print("TEST :", df_test["Date"].min(),  "->", df_test["Date"].max(),  "rows:", len(df_test))

    # Arrays and scaling
    arr_train = df_to_arr(df_train)
    arr_val   = df_to_arr(df_val)
    arr_test  = df_to_arr(df_test)

    train_s, val_s, test_s, train_mean, train_std = scale_with_train_stats(arr_train, arr_val, arr_test)
    close_mean, close_std = train_mean[0,3], train_std[0,3]
    unscale = lambda v: (v * close_std) + close_mean

    # Context
    CONTEXT_LEN = int(CONTEXT_LEN_FIXED)

    # Sequences
    X_train, y_train = make_sequences(train_s, CONTEXT_LEN)
    X_val,   y_val   = make_sequences(val_s,   CONTEXT_LEN)
    X_test,  y_test  = make_sequences(test_s,  CONTEXT_LEN)

    # Optional: quick sanity check for alignment (first 5 windows)
    print("Sanity check window->target dates (first 5):")
# BROKEN:     for i in range(5):
        win_end_idx = i + CONTEXT_LEN - 1
        target_idx = i + CONTEXT_LEN
        print(i, "window_end_date:", df_test.loc[win_end_idx, "Date"], "target_date:", df_test.loc[target_idx, "Date"])

    # Pretraining
    pre_hist = None
# BROKEN:     if PRETRAIN_ENABLED:
        mae_model = build_masked_autoencoder(CONTEXT_LEN)
        pre_hist = pretrain_autoencoder(mae_model, X_train, epochs=EPOCHS_PRE)

    # Fine-tune forecaster with clipped sigma and auxiliary MAE
    X_ft = np.concatenate([X_train, X_val], axis=0)
    y_ft = np.concatenate([y_train, y_val], axis=0)
    forecaster = build_prob_forecaster_clipped(CONTEXT_LEN)
    ft_hist = finetune_prob_forecaster_aux(forecaster, X_ft, y_ft, epochs=EPOCHS_FT)

    # Evaluate on test (scaled)
    preds = forecaster.predict(X_test, verbose=0)
    mu    = preds[:, 0:1]
    sigma = np.exp(preds[:, 1:2])  # note: log_sigma clipped before exp
    pers  = persistence_pred_scaled(X_test)

    mae_model_scaled = mae(mu, y_test)
    mae_pers_scaled  = mae(pers, y_test)
    e_model = np.abs(y_test - mu).flatten()
    e_pers  = np.abs(y_test - pers).flatten()
    dm_stat = dm_test(e_model, e_pers)

    # Original scale + calibration
    mae_model_orig = mae(unscale(mu),   unscale(y_test))
    mae_pers_orig  = mae(unscale(pers), unscale(y_test))
    coverage = calibration_coverage(mu, sigma, y_test, bands=(1.0, 2.0))

    # Diagnostic dumps (first 20)
    print("Diagnostic dump (first 20):")
# BROKEN:     for i in range(min(20, len(mu))):
        print(i,
            "mu", float(mu[i]),
            "sigma", float(sigma[i]),
            "y", float(y_test[i]),
            "pers", float(pers[i]),
            "mu_un", float(unscale(mu[i])),
            "y_un", float(unscale(y_test[i])),
            "pers_un", float(unscale(pers[i])))

    print("mu_bias_scaled:", float(np.mean(mu - y_test)))
    print("mu_rmse_scaled:", float(np.sqrt(np.mean((mu - y_test)**2))))
    print("sigma_mean:", float(np.mean(sigma)), "sigma_std:", float(np.std(sigma)))

    # Manifest
    manifest = {
        "run_id": int(time.time()),
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "script": "hybrid_fisheye_prob_ann_patch_run.py",
        "run_name": RUN_NAME,
        "seed": SEED,
        "flags": {
            "pretrain_enabled": PRETRAIN_ENABLED,
            "context_len": CONTEXT_LEN,
            "lr_pre": LEARN_RATE_PRE,
            "lr_ft": LEARN_RATE_FT,
            "dropout": DROPOUT,
            "l2": L2W,
            "mask_ratio": MASK_RATIO,
            "mlp_widths": MLP_WIDTHS,
            "aux_mae_alpha": AUX_MAE_ALPHA,
            "logsigma_clip": [LOGSIGMA_CLIP_MIN, LOGSIGMA_CLIP_MAX]
        },
        "files": {
            "train_csv": os.path.abspath(TRAIN_CSV),
            "val_csv": os.path.abspath(VAL_CSV),
            "test_csv": os.path.abspath(TEST_CSV),
            "train_sha256": sha256_file(TRAIN_CSV),
            "val_sha256": sha256_file(VAL_CSV),
            "test_sha256": sha256_file(TEST_CSV)
        },
        "date_ranges": {
            "train_min": str(df_train["Date"].min()), "train_max": str(df_train["Date"].max()),
            "val_min": str(df_val["Date"].min()),     "val_max": str(df_val["Date"].max()),
            "test_min": str(df_test["Date"].min()),   "test_max": str(df_test["Date"].max())
        },
        "metrics_scaled": {
            "mae_model": mae_model_scaled,
            "mae_persistence": mae_pers_scaled,
            "dm_stat_abs_error": dm_stat
        },
        "metrics_original": {
            "mae_model": mae_model_orig,
            "mae_persistence": mae_pers_orig
        },
        "calibration": coverage,
        "pretrain": {
            "enabled": PRETRAIN_ENABLED,
            "last_val_loss": None if (pre_hist is None) else float(pre_hist.history.get("val_loss", [np.nan])[-1]),
            "epochs_run": None if (pre_hist is None) else len(pre_hist.history.get("loss", []))
        },
        "finetune": {
            "last_val_loss": float(ft_hist.history.get("val_loss", [np.nan])[-1]),
            "epochs_run": len(ft_hist.history.get("loss", []))
        },
        "closure": "sealed"
    }
    out_manifest = f"{RUN_NAME}_manifest.json"
    write_manifest(manifest, out_manifest)

    # Summary prints
    print(f"{RUN_NAME}: complete. Manifest written: {out_manifest}")
    print({
        "run": RUN_NAME,
        "scaled": {
            "mae_model": mae_model_scaled,
            "mae_persistence": mae_pers_scaled,
            "dm_stat_abs_error": dm_stat
        },
        "original": {
            "mae_model": mae_model_orig,
            "mae_persistence": mae_pers_orig
        },
        "calibration": coverage
    })
    print(f"Manifest: {out_manifest}")
