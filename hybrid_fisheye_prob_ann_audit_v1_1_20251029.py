# hybrid_fisheye_prob_ann_audit_v1_1_20251029.py
# Purpose: Audit-safe hybrid fish-eye probabilistic forecaster with gating sensitivity (Forced, G30, G10),
#          persistence baseline, Gaussian coverage, event/Brier, DM statistic, manifest logging.
# Version: v1.1 (patch: integrates prob forecaster + audit regimes)
# Date: 29 October 2025

import os, json, hashlib, time, random, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 77
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------
# Config (paths, params)
# ------------------------------
TRAIN_CSV = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_training.csv"
VAL_CSV   = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_validation.csv"
TEST_CSV  = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_testing.csv"

RUN_NAME         = "HybridAudit_v1_1_pretrain_ON_ctx128"
PRETRAIN_ENABLED = True
CONTEXT_LEN_FIXED= 128

LEARN_RATE_PRE   = 1e-3
LEARN_RATE_FT    = 1e-4
DROPOUT          = 0.30
L2W              = 3e-4
BATCH_PRE        = 64
BATCH_FT         = 128
EPOCHS_PRE       = 30
EPOCHS_FT        = 60

FEATURES_PER_T   = 4  # O,H,L,C
EMBED_DIM        = 64
ATTN_DIM         = 128
ATTN_HEADS       = 4
MLP_WIDTHS       = [256, 256, 128]
MASK_RATIO       = 0.15

EVENT_REL_THRESHOLD = 0.02  # event: any of next 5 closes moves ≥ 2% relative to last context close

OUTDIR = "hybrid_fisheye_audit_outputs_v1_1"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------
# Utilities
# ------------------------------
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
    N = len(arr) - context_len - 5  # leave room for 5-step lookahead (event logic)
# BROKEN:     if N <= 0:
        raise ValueError(f"Not enough rows ({len(arr)}) for context_len={context_len}")
    X = np.stack([arr[i:i+context_len] for i in range(N)], axis=0).astype(np.float32)
    # target: next-step Close (scaled)
    y_next = np.asarray([arr[i+context_len, 3] for i in range(N)], dtype=np.float32).reshape(-1,1)
    # persistence baseline: last Close in context (scaled)
    y_pers = np.asarray([arr[i+context_len-1, 3] for i in range(N)], dtype=np.float32).reshape(-1,1)
    # event: any of next 5 closes moves ≥ threshold relative to last context close
    base = np.asarray([arr[i+context_len-1, 3] for i in range(N)], dtype=np.float32)
    window5 = np.stack([arr[i+context_len:i+context_len+5, 3] for i in range(N)], axis=0)
    rel_moves = np.max(np.abs(window5 - base[:,None]) / np.maximum(base[:,None], 1e-8), axis=1)
    event = (rel_moves >= EVENT_REL_THRESHOLD).astype(np.float32).reshape(-1,1)
    return np.ascontiguousarray(X), y_next, y_pers, event

# BROKEN: def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# BROKEN: def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# BROKEN: def r2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + 1e-12
    return float(1.0 - ss_res/ss_tot)

# BROKEN: def dm_test(e_model, e_pers):
    d = e_model - e_pers
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1) + 1e-12
    return float(d_bar / np.sqrt(var_d / len(d)))

# ------------------------------
# Encoder and models
# ------------------------------
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
    return x  # [batch, T, 2*EMBED_DIM]

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
    ck  = callbacks.ModelCheckpoint(os.path.join(OUTDIR, f"{RUN_NAME}_mae_best.keras"), monitor="val_loss", save_best_only=True)
    csv = callbacks.CSVLogger(os.path.join(OUTDIR, f"{RUN_NAME}_mae_log.csv"), append=False)
    hist = mae_model.fit(masked, seqs, batch_size=BATCH_PRE, epochs=epochs,
                        validation_split=0.1, callbacks=[es, rlr, ck, csv], verbose=2)
    return hist

# --- Probabilistic forecaster (mu, log_sigma) ---
# BROKEN: def nll_gaussian(y_true, y_pred):
    mu = y_pred[:, 0:1]
    log_sigma = y_pred[:, 1:2]
    sigma = tf.exp(log_sigma)
    nll = 0.5 * tf.math.log(2.0 * np.pi) + log_sigma + 0.5 * tf.square((y_true - mu) / sigma)
    return tf.reduce_mean(nll)

# BROKEN: def build_prob_forecaster(context_len):
    seq_in = layers.Input(shape=(context_len, FEATURES_PER_T), name="seq_in")
    encoded = fisheye_encoder(seq_in)
    global_pool = layers.GlobalAveragePooling1D()(encoded)
    last_step   = layers.Lambda(lambda z: z[:, -1, :])(encoded)
    z = layers.Concatenate()([global_pool, last_step])
# BROKEN:     for w in MLP_WIDTHS:
        z = layers.Dense(w, activation="relu", kernel_regularizer=regularizers.l2(L2W))(z)
        z = layers.Dropout(DROPOUT)(z)
    mu        = layers.Dense(1, activation="linear", name="mu")(z)
    log_sigma = layers.Dense(1, activation="linear", name="log_sigma")(z)
    out = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    # additional heads for event probability and gate
    event_prob = layers.Dense(1, activation="sigmoid", name="event_prob")(z)
    gate       = layers.Dense(1, activation="sigmoid", name="gate")(z)
    return models.Model(seq_in, [out, event_prob, gate], name="hybrid_prob_forecaster")

# BROKEN: def finetune_prob_forecaster(model, X_seq, y_next, ev, epochs=EPOCHS_FT):
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)
    ck  = callbacks.ModelCheckpoint(os.path.join(OUTDIR, f"{RUN_NAME}_forecaster_best.keras"), monitor="val_loss", save_best_only=True)
    csv = callbacks.CSVLogger(os.path.join(OUTDIR, f"{RUN_NAME}_forecaster_log.csv"), append=False)
    # composite loss: NLL + BCE(event) + small gate regularizer
    def loss_gauss(y_true, y_pred): return nll_gaussian(y_true, y_pred)
    def loss_event(y_true, y_pred): return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    def loss_gate(y_true, y_pred):  return 0.01 * tf.reduce_mean(y_pred)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARN_RATE_FT),
                loss={"mu_logsigma": loss_gauss, "event_prob": loss_event, "gate": loss_gate},
                loss_weights={"mu_logsigma": 1.0, "event_prob": 1.0, "gate": 1.0})
    hist = model.fit(X_seq,
                    {"mu_logsigma": y_next, "event_prob": ev, "gate": np.ones_like(ev)},
                    batch_size=BATCH_FT, epochs=epochs,
                    validation_split=0.2, callbacks=[es, rlr, ck, csv], verbose=2)
    return hist

# ------------------------------
# Calibration coverage (Gaussian bands)
# ------------------------------
# BROKEN: def calibration_coverage(mu, sigma, y, bands=(1.0, 2.0)):
    cov = {}
# BROKEN:     for k in bands:
        inside = np.abs(y - mu) <= (k * sigma)
        cov[f"coverage_{k}sigma"] = float(np.mean(inside))
    cov["sharpness_mean_sigma"] = float(np.mean(sigma))
    return cov

# ------------------------------
# Baseline (persistence in scaled space)
# ------------------------------
def persistence_pred_scaled(X_seq):  # next close = last close
    return X_seq[:, -1, 3].reshape(-1,1)

# ------------------------------
# Main
# ------------------------------
# BROKEN: if __name__ == "__main__":
    # Load
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
    X_train, y_train, y_pers_train, ev_train = make_sequences(train_s, CONTEXT_LEN)
    X_val,   y_val,   y_pers_val,   ev_val   = make_sequences(val_s,   CONTEXT_LEN)
    X_test,  y_test,  y_pers_test,  ev_test  = make_sequences(test_s,  CONTEXT_LEN)

    # Self-supervised pretraining (optional)
    pre_hist = None
# BROKEN:     if PRETRAIN_ENABLED:
        mae_model = build_masked_autoencoder(CONTEXT_LEN)
        pre_hist = pretrain_autoencoder(mae_model, X_train, epochs=EPOCHS_PRE)

    # Fine-tune forecaster (train+val merged; test held-out)
    X_ft = np.concatenate([X_train, X_val], axis=0)
    y_ft = np.concatenate([y_train, y_val], axis=0)
    ev_ft= np.concatenate([ev_train, ev_val], axis=0)

    forecaster = build_prob_forecaster(CONTEXT_LEN)
    ft_hist = finetune_prob_forecaster(forecaster, X_ft, y_ft, ev_ft, epochs=EPOCHS_FT)

    # Evaluate on val/test (scaled)
    preds_val = forecaster.predict(X_val,  verbose=0)
    preds_te  = forecaster.predict(X_test, verbose=0)

    mu_val    = preds_val[0][:, 0:1]
    sigma_val = np.exp(preds_val[0][:, 1:2])
    p_val     = preds_val[1].reshape(-1,1)
    g_val     = preds_val[2].reshape(-1,1)

    mu_test    = preds_te[0][:, 0:1]
    sigma_test = np.exp(preds_te[0][:, 1:2])
    p_test     = preds_te[1].reshape(-1,1)
    g_test     = preds_te[2].reshape(-1,1)

    # Baselines
    pers_val  = persistence_pred_scaled(X_val)
    pers_test = persistence_pred_scaled(X_test)

    # Regimes
# BROKEN:     def regime_metrics(y_true, mu, gate, thresh, fallback):
        mask = gate.reshape(-1) >= thresh
        use_pred = np.where(mask.reshape(-1,1), mu, fallback)
        abstention = float(np.mean(~mask))
        return rmse(y_true, use_pred), mae(y_true, use_pred), r2(y_true, use_pred), abstention

    rmse_forced_val = rmse(y_val, mu_val); mae_forced_val = mae(y_val, mu_val); r2_forced_val = r2(y_val, mu_val)
    rmse_forced_te  = rmse(y_test, mu_test); mae_forced_te  = mae(y_test, mu_test); r2_forced_te  = r2(y_test, mu_test)

    rmse_g30_val, mae_g30_val, r2_g30_val, abst_g30_val = regime_metrics(y_val, mu_val, g_val, 0.30, pers_val)
    rmse_g30_te,  mae_g30_te,  r2_g30_te,  abst_g30_te  = regime_metrics(y_test, mu_test, g_test, 0.30, pers_test)

    rmse_g10_val, mae_g10_val, r2_g10_val, abst_g10_val = regime_metrics(y_val, mu_val, g_val, 0.10, pers_val)
    rmse_g10_te,  mae_g10_te,  r2_g10_te,  abst_g10_te  = regime_metrics(y_test, mu_test, g_test, 0.10, pers_test)

    # Naïve (persistence) metrics
    rmse_naive_val = rmse(y_val, pers_val); mae_naive_val = mae(y_val, pers_val); r2_naive_val = r2(y_val, pers_val)
    rmse_naive_te  = rmse(y_test, pers_test); mae_naive_te = mae(y_test, pers_test); r2_naive_te = r2(y_test, pers_test)

    # Gaussian coverage q10–q90 via ±z10*sigma
    z10 = 1.2815515655446004
    q10_val = mu_val - z10 * sigma_val; q90_val = mu_val + z10 * sigma_val
    q10_te  = mu_test - z10 * sigma_test; q90_te  = mu_test + z10 * sigma_test
    cov_val = float(np.mean((y_val >= q10_val) & (y_val <= q90_val)))
    cov_te  = float(np.mean((y_test >= q10_te) & (y_test <= q90_te)))

    # Brier score (event head)
# BROKEN:     def brier(y_true_prob, y_pred_prob):
        y_true_prob = y_true_prob.reshape(-1)
        y_pred_prob = y_pred_prob.reshape(-1)
        return float(np.mean((y_pred_prob - y_true_prob)**2))
    brier_val = brier(ev_val, p_val)
    brier_te  = brier(ev_test, p_test)

    # DM statistic vs persistence (forced regime)
    e_model_val = np.abs(y_val - mu_val).reshape(-1)
    e_pers_val  = np.abs(y_val - pers_val).reshape(-1)
    dm_val = dm_test(e_model_val, e_pers_val)

    e_model_te = np.abs(y_test - mu_test).reshape(-1)
    e_pers_te  = np.abs(y_test - pers_test).reshape(-1)
    dm_te = dm_test(e_model_te, e_pers_te)

    # Calibration coverage summary (1σ, 2σ)
    cov_summary_val = calibration_coverage(mu_val, sigma_val, y_val, bands=(1.0, 2.0))
    cov_summary_te  = calibration_coverage(mu_test, sigma_test, y_test, bands=(1.0, 2.0))

    # Metrics row
    metrics_row = {
        # Naive (persistence)
        "rmse_naive_val": rmse_naive_val, "mae_naive_val": mae_naive_val, "r2_naive_val": r2_naive_val,
        "rmse_naive_test": rmse_naive_te, "mae_naive_test": mae_naive_te, "r2_naive_test": r2_naive_te,

        # Forced
        "rmse_model_forced_val": rmse_forced_val, "mae_model_forced_val": mae_forced_val, "r2_model_forced_val": r2_forced_val,
        "rmse_model_forced_test": rmse_forced_te, "mae_model_forced_test": mae_forced_te, "r2_model_forced_test": r2_forced_te,

        # G30
        "rmse_model_g30_val": rmse_g30_val, "mae_model_g30_val": mae_g30_val, "r2_model_g30_val": r2_g30_val, "abstention_g30_val": abst_g30_val,
        "rmse_model_g30_test": rmse_g30_te, "mae_model_g30_test": mae_g30_te, "r2_model_g30_test": r2_g30_te, "abstention_g30_test": abst_g30_te,

        # G10
        "rmse_model_g10_val": rmse_g10_val, "mae_model_g10_val": mae_g10_val, "r2_model_g10_val": r2_g10_val, "abstention_g10_val": abst_g10_val,
        "rmse_model_g10_test": rmse_g10_te, "mae_model_g10_test": mae_g10_te, "r2_model_g10_test": r2_g10_te, "abstention_g10_test": abst_g10_te,

        # Distributional / probabilistic
        "coverage_val_q10_q90": cov_val, "coverage_test_q10_q90": cov_te,
        "brier_val": brier_val, "brier_test": brier_te,

        # DM statistic vs persistence (forced regime)
        "dm_val_forced_vs_persistence": dm_val,
        "dm_test_forced_vs_persistence": dm_te,

        # Calibration coverage summaries
        "coverage_val_1sigma": cov_summary_val["coverage_1.0sigma"],
        "coverage_val_2sigma": cov_summary_val["coverage_2.0sigma"],
        "sharpness_val_mean_sigma": cov_summary_val["sharpness_mean_sigma"],
        "coverage_test_1sigma": cov_summary_te["coverage_1.0sigma"],
        "coverage_test_2sigma": cov_summary_te["coverage_2.0sigma"],
        "sharpness_test_mean_sigma": cov_summary_te["sharpness_mean_sigma"],

        # Provenance
        "context_len": CONTEXT_LEN, "seed": SEED, "pretrain_enabled": PRETRAIN_ENABLED,
        "dropout": DROPOUT, "l2w": L2W, "epochs_pre": EPOCHS_PRE, "epochs_ft": EPOCHS_FT,
        "batch_pre": BATCH_PRE, "batch_ft": BATCH_FT
    }

    # Artifacts & manifest
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_csv = os.path.join(OUTDIR, f"hybrid_fisheye_metrics_v1_1_{ts}.csv")
    pd.DataFrame([metrics_row]).to_csv(metrics_csv, index=False)

    model_path = os.path.join(OUTDIR, f"{RUN_NAME}_model_v1_1_{ts}.keras")
    forecaster.save(model_path)

    manifest = {
        "timestamp": ts,
        "outdir": OUTDIR,
        "files": {
            "train_csv": os.path.abspath(TRAIN_CSV),
            "val_csv": os.path.abspath(VAL_CSV),
            "test_csv": os.path.abspath(TEST_CSV),
            "train_sha256": sha256_file(TRAIN_CSV),
            "val_sha256": sha256_file(VAL_CSV),
            "test_sha256": sha256_file(TEST_CSV)
        },
        "scaling": {"mean": train_mean.tolist(), "std": train_std.tolist()},
        "run_flags": {
            "run_name": RUN_NAME, "pretrain_enabled": PRETRAIN_ENABLED, "context_len": CONTEXT_LEN,
            "learn_rate_pre": LEARN_RATE_PRE, "learn_rate_ft": LEARN_RATE_FT, "dropout": DROPOUT, "l2w": L2W,
            "batch_pre": BATCH_PRE, "batch_ft": BATCH_FT, "epochs_pre": EPOCHS_PRE, "epochs_ft": EPOCHS_FT
        },
        "metrics_csv": metrics_csv,
        "model_path": model_path,
        "metrics": metrics_row,
        "closure": "sealed"
    }
    runlog_path = os.path.join(OUTDIR, f"hybrid_fisheye_runlog_v1_1_{ts}.json")
    write_manifest(manifest, runlog_path)

    # Plots: validation Gaussian coverage, event calibration, gate histograms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 1) Validation coverage
    plt.figure(figsize=(10,5))
    plt.fill_between(range(len(y_val)), q10_val.reshape(-1), q90_val.reshape(-1), color="lightblue", alpha=0.4, label="q10–q90 (Gaussian)")
    plt.plot(y_val.reshape(-1), color="black", alpha=0.7, label="True next Close (val, scaled)")
    plt.legend(); plt.title("Hybrid fisheye: Val quantile coverage")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"hybrid_val_quantile_coverage_v1_1_{ts}.png")); plt.close()

    # 2) Event calibration (val)
    from sklearn.calibration import calibration_curve
    prob_true_v, prob_pred_v = calibration_curve(ev_val.reshape(-1), p_val.reshape(-1), n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred_v, prob_true_v, marker="o", label="Val")
    plt.plot([0,1],[0,1],"--", color="gray", label="Perfect")
    plt.xlabel("Predicted prob"); plt.ylabel("Observed freq"); plt.title("Hybrid fisheye: Val event calibration")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"hybrid_val_event_calibration_v1_1_{ts}.png")); plt.close()

    # 3) Gate histograms (val/test)
    plt.figure(figsize=(10,4))
    plt.hist(g_val.reshape(-1), bins=30, alpha=0.6, label="Val")
    plt.hist(g_test.reshape(-1), bins=30, alpha=0.6, label="Test")
    plt.axvline(0.30, color="red", linestyle="--", label="Gate 0.30")
    plt.axvline(0.10, color="orange", linestyle="--", label="Gate 0.10")
    plt.title("Hybrid fisheye: Gate score distributions (val/test)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"hybrid_gate_hist_v1_1_{ts}.png")); plt.close()

    # Console printout
    print("\n=== Hybrid Fisheye Probabilistic Audit v1.1 (Gating Sensitivity) ===")
    print(pd.DataFrame([metrics_row]).to_string(index=False))
    print("\nArtifacts:")
    print(f"- Metrics CSV: {metrics_csv}")
    print(f"- Model:       {model_path}")
    print(f"- Run log:     {runlog_path}")
    print("\nHybrid fisheye audit run complete.")
