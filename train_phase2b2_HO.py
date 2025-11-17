#!/usr/bin/env python3
"""
train_phase2b2_HO.py
CANONICAL_NAME = "train_phase2b2_HO"
2bANN.2 Hybrid Architecture - Heating Oil (HO) dataset
Audit-safe training script with explicit train/val/test splits
Produces: scaler.pkl, scaler_sidecar.json when --scaler_only; full training writes RUNLOG JSON and artifacts
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
import argparse
import pickle
import sys

# --- Data loading ---
def load_ho_csv(path):
    colnames = ["Date", "Open", "High", "Low", "Close"]
    df = pd.read_csv(
        path,
        sep=None,             # auto-detect delimiter
        engine="python",      # required for sep=None
        header=0,
        names=colnames,
        parse_dates=['Date']
    )
    dates = df['Date'].values
    # keep numeric features used by pipeline
    X = df[['Open', 'High', 'Low', 'Close']].values.astype(float)
    y = df['Close'].values.astype(float)
    print(f"{os.path.basename(path)} -> dates: {dates.shape}, X: {X.shape}, y: {y.shape}")
    return dates, X, y

# --- Evaluation helpers ---
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    return mae, rmse, r2

def plot_predictions(dates, y_true, y_pred, title="ANN Predictions vs Actual"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
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
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def log_run(meta, outdir):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = os.path.join(outdir, f"RUNLOG_2bANN2_HO_{ts}.json")
    with open(log_name, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[LOGGED] {log_name}")
    return log_name

# --- Model builder (simple MLP) ---
def build_model(input_dim, hidden_units=(64,128,64), dropout=0.1, l2_reg=0.0):
    inp = layers.Input(shape=(input_dim,), name="features")
    x = inp
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                         name=f"dense_{i+1}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)
    out = layers.Dense(1, activation='linear', name="regression_output")(x)
    model = keras.Model(inputs=inp, outputs=out, name="2bANN2_HO_model")
    return model

# --- Main routine ---
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="HO train_phase2b2_HO model runner (deterministic, auditable)")
    p.add_argument("--ho", required=False, default=os.getcwd(), help="HO working folder")
    p.add_argument("--artout", required=False, default=os.path.join(os.getcwd(), "ho_artifact_outputs"), help="artifact output folder")
    p.add_argument("--scaler_only", action="store_true", help="only produce scaler.pkl and sidecar")
    p.add_argument("--seed", type=int, default=20251030, help="deterministic seed")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--hidden_units", nargs="+", type=int, default=[64,128,64])
    args = p.parse_args(argv)

    # normalize paths and ensure out dirs
    args.ho = os.path.abspath(args.ho)
    args.artout = os.path.abspath(args.artout)
    os.makedirs(args.ho, exist_ok=True)
    os.makedirs(args.artout, exist_ok=True)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # dataset paths (using args.ho as requested)
    train_path = os.path.join(args.ho, "hoxnc_training.csv")
    val_path   = os.path.join(args.ho, "hoxnc_validation.csv")
    test_path  = os.path.join(args.ho, "hoxnc_testing.csv")

    # load datasets (if missing, will raise; scaler-only expects training file)
    dates_train, X_train, y_train = load_ho_csv(train_path)
    dates_val, X_val, y_val = load_ho_csv(val_path)
    dates_test, X_test, y_test = load_ho_csv(test_path)

    # --- Scaling ---
    scaler = StandardScaler()
    scaler.fit(X_train)

    scaler_path = os.path.join(args.ho, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    sidecar = {
        "feature_names": ["Date","Open","High","Low","Close"],
        "means": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "scales": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "scaler_class": type(scaler).__name__,
        "saved_at": datetime.datetime.now().isoformat()
    }
    sidecar_path = os.path.join(args.ho, "scaler_sidecar.json")
    with open(sidecar_path, "w", encoding="utf8") as f:
        json.dump(sidecar, f, indent=2, default=str)

    print("WROTE", scaler_path)
    print("WROTE", sidecar_path)

    if args.scaler_only:
        print("SCALER_ONLY_DONE")
        return 0

    # apply scaler
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)

    # deterministic split / training
    model = build_model(input_dim=Xs_train.shape[1], hidden_units=tuple(args.hidden_units), dropout=args.dropout, l2_reg=args.l2)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss="mean_absolute_error", metrics=["mean_absolute_error"])

    ckpt_path = os.path.join(args.ho, "2bANN2_HO_model.keras")
    cb_checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=False)
    cb_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)

    history = model.fit(
        Xs_train, y_train,
        validation_data=(Xs_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[cb_checkpoint, cb_early],
        verbose=2
    )

    # final eval
    val_pred = model.predict(Xs_val, batch_size=args.batch_size).squeeze()
    mae, rmse, r2 = evaluate_predictions(y_val, val_pred)

    runmeta = {
        "run_started": datetime.datetime.now().isoformat(),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": ckpt_path,
        "scaler_path": scaler_path,
        "artifact_tag": "2bANN.2_HO",
        "notes": "Hybrid regression HO run"
    }
    runlog_path = log_run(runmeta, args.artout)

    # deterministic inference hooks - placeholders (deterministic_inference.py should be called here)
    det_null_target = os.path.join(args.artout, "deterministic_inference_result_null_target.json")
    det_null_features = os.path.join(args.artout, "deterministic_inference_result_null_features.json")
    with open(det_null_target, "w", encoding="utf8") as f:
        json.dump({"note": "null_target_instrumentation", "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)
    with open(det_null_features, "w", encoding="utf8") as f:
        json.dump({"note": "null_features_instrumentation", "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)

    # compute artifact hashes
    artifacts = {}
    for p in (runlog_path, det_null_target, det_null_features, ckpt_path, scaler_path, sidecar_path):
        if os.path.exists(p):
            artifacts[os.path.basename(p)] = sha256_file(p)
    artifact_summary_path = os.path.join(args.artout, "artifact_hashes_summary.json")
    with open(artifact_summary_path, "w", encoding="utf8") as f:
        json.dump(artifacts, f, indent=2)
    print("WROTE", artifact_summary_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())

