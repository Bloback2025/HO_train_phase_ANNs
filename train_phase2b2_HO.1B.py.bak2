#!/usr/bin/env python3
"""
train_phase2b2_HO.1B.py
Development variant of the Phase2b2 HO pipeline (higher-dev). Keeps audit-grade outputs
and all SHA sidecars; adds more robust training controls: scaling, early stopping,
L2 regularisation, and explicit RNG control for reproducibility.

Usage examples:
  python train_phase2b2_HO.1B.py --mode synthetic
  python train_phase2b2_HO.1B.py --mode live --train-csv path/to/train.csv --test-csv path/to/test.csv --target target_column
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks

# -----------------------
# Deterministic helpers
# -----------------------
def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# -----------------------
# Data generation and loading
# -----------------------
def make_synthetic(seed=20251117, n_samples=2000, n_features=10, noise=1.0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    coefs = np.linspace(1.0, 0.1, n_features)
    y = X.dot(coefs) + rng.normal(scale=noise, size=n_samples)
    cols = [f"f{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

def make_random(seed=20251117, n_samples=2000, n_features=10):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    cols = [f"f{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -----------------------
# Metrics and model
# -----------------------
def compute_metrics(y_true, y_pred):
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    residuals = (np.array(y_true) - np.array(y_pred))
    residual_mean = float(np.mean(residuals))
    return {"r2": r2, "mae": mae, "rmse": rmse, "residual_mean": residual_mean, "n_test": int(len(y_true))}

def build_keras_model(input_dim: int, seed=20251117, l2=1e-4):
    tf_seed = seed % (2**31 - 1)
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dense(1, activation="linear")
    ])
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mse", metrics=[])
    return model

# -----------------------
# Preflight checks
# -----------------------
def preflight_checks(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):
    errors = []
    if target not in df_train.columns:
        errors.append(f"target '{target}' not in train columns")
    if target not in df_test.columns:
        errors.append(f"target '{target}' not in test columns")
    # lightweight duplicate detection (string-joined feature rows)
    feats_train = df_train.drop(columns=[target]).astype(str).agg("|".join, axis=1)
    feats_test = df_test.drop(columns=[target]).astype(str).agg("|".join, axis=1)
    dup = set(feats_train).intersection(set(feats_test))
    if len(dup) > 0:
        errors.append(f"{len(dup)} duplicate feature-only rows found across train/test")
    return errors

# -----------------------
# Main run logic
# -----------------------
def run(args):
    seed = int(args.seed)
    run_id = args.run_id
    outdir = Path("ho_artifact_outputs") / run_id
    ensure_dir(outdir)

    # deterministic seeds for numpy and tensorflow
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        tf = None

    # Acquire or generate data
    if args.mode == "synthetic":
        df = make_synthetic(seed=seed, n_samples=args.n_samples, n_features=args.n_features, noise=args.noise)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    elif args.mode == "random":
        df = make_random(seed=seed, n_samples=args.n_samples, n_features=args.n_features)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    elif args.mode == "live":
        if not args.train_csv or not args.test_csv:
            print("ERROR: live mode requires --train-csv and --test-csv", file=sys.stderr)
            sys.exit(2)
        df_train = load_csv(args.train_csv)
        df_test = load_csv(args.test_csv)
    else:
        print("ERROR: unknown mode", args.mode, file=sys.stderr)
        sys.exit(2)

    # Preflight
    errors = preflight_checks(df_train, df_test, args.target)
    if errors:
        print("PREFLIGHT_FAIL:")
        for e in errors:
            print(" -", e)
        sys.exit(3)

    # Prepare matrices and scale
    X_train = df_train.drop(columns=[args.target]).values
    y_train = df_train[args.target].values
    X_test = df_test.drop(columns=[args.target]).values
    y_test = df_test[args.target].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with Keras and early stopping; guarded fallback to linear
    model_used = None
    preds = None
    try:
        if tf is None:
            raise RuntimeError("TensorFlow not available")
        model = build_keras_model(input_dim=X_train_scaled.shape[1], seed=seed, l2=args.l2)
        es = callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=0)
        model.fit(X_train_scaled, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0, callbacks=[es])
        preds = model.predict(X_test_scaled, batch_size=args.batch_size).reshape(-1)
        model_used = "Keras_MLP"
        # save weights to outdir for traceability
        weights_path = str(outdir / "model_weights.h5")
        model.save_weights(weights_path)
        weights_sha = sha256_of_file(weights_path)
        with open(weights_path + ".sha256.txt", "w", encoding="ascii") as f:
            f.write(weights_sha)
    except Exception:
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        preds = lr.predict(X_test_scaled)
        model_used = "LinearRegression_fallback"

    metrics = compute_metrics(y_test, preds)

    # Write preds_model.json
    preds_out = [{"index": int(i), "prediction": float(p)} for i, p in enumerate(preds)]
    preds_path = outdir / "preds_model.json"
    write_json_atomic(preds_out, str(preds_path))

    # Write promotion_gate_summary.json
    script_name = str(Path(args.script_path).name)
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_used": model_used,
        "seed": seed,
        "metrics": metrics,
        "n_features": int(X_test.shape[1]),
        "created_at": now_iso(),
        "script": {"path": script_name, "sha256": None}
    }
    summary_path = outdir / "promotion_gate_summary.json"
    write_json_atomic(summary, str(summary_path))

    # Persist scaler for reproducibility
    try:
        import joblib
        scaler_path = outdir / "scaler.joblib"
        joblib.dump(scaler, str(scaler_path))
        scaler_sha = sha256_of_file(str(scaler_path))
        with open(str(scaler_path) + ".sha256.txt", "w", encoding="ascii") as f:
            f.write(scaler_sha)
    except Exception:
        scaler_sha = None

    # Compute SHA sidecars for preds and summary
    preds_sha = sha256_of_file(str(preds_path))
    summary_sha = sha256_of_file(str(summary_path))
    with open(str(preds_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(preds_sha)
    with open(str(summary_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(summary_sha)

    # Compute script SHA and write sidecar
    script_path = Path(args.script_path).resolve()
    script_sha = sha256_of_file(str(script_path))
    with open(str(script_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(script_sha)

    # Update summary with script sha and rewrite
    summary["script"]["sha256"] = script_sha
    write_json_atomic(summary, str(summary_path))
    summary_sha = sha256_of_file(str(summary_path))
    with open(str(summary_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(summary_sha)

    # Create local manifest patch in outdir (non-destructive)
    manifest_patch = {
        "run_id": run_id,
        "manifest_patched_at": now_iso(),
        "scripts": {"promoted": {"path": script_name, "sha256": script_sha}},
        "outputs": {
            "preds_model": {"path": str(preds_path), "sha256": preds_sha},
            "promotion_gate": {"path": str(summary_path), "sha256": summary_sha},
            "scaler": {"path": str(scaler_path) if 'scaler_path' in locals() else None, "sha256": scaler_sha}
        }
    }
    manifest_path = outdir / f"run_manifest.{run_id}.patched.json"
    write_json_atomic(manifest_patch, str(manifest_path))
    manifest_sha = sha256_of_file(str(manifest_path))
    with open(str(manifest_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(manifest_sha)

    # Optionally append closure to known repo audit_summary if present (non-destructive)
    candidate_audit = Path("C:/Users/loweb/AI_Financial_Sims/HO/HO 1st time 5080/audit_summary_for_notepad.txt")
    if candidate_audit.exists():
        closure = ("--- RUN_CLOSED: {run_id} ; SCRIPT: {script} ; SUMMARY_SHA: {sha} ; TIMESTAMP: {ts} ---"
                   .format(run_id=run_id, script=script_name, sha=summary_sha, ts=now_iso()))
        with open(candidate_audit, "a", encoding="utf8") as f:
            f.write("\n" + closure + "\n")

    # Print audit verification lines
    print("WROTE:", str(preds_path))
    print("WROTE:", str(summary_path))
    print("SCRIPT_SHA:", script_sha)
    print("PREDS_SHA:", preds_sha)
    print("SUMMARY_SHA:", summary_sha)
    print("MANIFEST_PATCHED:", str(manifest_path))
    print("MANIFEST_SHA:", manifest_sha)
    print("OUTDIR:", str(outdir))

# -----------------------
# CLI
# -----------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="train_phase2b2_HO.1B.py - development train pipeline")
    p.add_argument("--mode", choices=["synthetic", "random", "live"], default="synthetic")
    p.add_argument("--run-id", dest="run_id", default="synthetic_forecast_test_20251117")
    p.add_argument("--seed", type=int, default=20251117)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-features", dest="n_features", type=int, default=10)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--train-csv", dest="train_csv", default=None)
    p.add_argument("--test-csv", dest="test_csv", default=None)
    p.add_argument("--target", dest="target", default="target")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--script-path", dest="script_path", default=__file__)
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args)
