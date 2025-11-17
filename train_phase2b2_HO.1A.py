#!/usr/bin/env python3
"""
train_phase2b2_HO.1A.py
Canonical train/inference pipeline (audit-grade).

Produces:
 - ho_artifact_outputs/<run_id>/preds_model.json
 - ho_artifact_outputs/<run_id>/promotion_gate_summary.json
 - SHA256 sidecars for produced artifacts and the script itself
 - run_manifest.<run_id>.patched.json (local manifest patch)

Usage examples:
  python train_phase2b2_HO.1A.py --mode synthetic
  python train_phase2b2_HO.1A.py --mode live --train-csv path/to/train.csv --test-csv path/to/test.csv --target target_column
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
from sklearn.neural_network import MLPRegressor

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

def train_model(X_train, y_train, seed=20251117, max_iter=500):
    model = MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation="relu",
                         solver="adam",
                         random_state=seed,
                         max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def fallback_linear(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

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

    # Prepare data matrices
    X_train = df_train.drop(columns=[args.target]).values
    y_train = df_train[args.target].values
    X_test = df_test.drop(columns=[args.target]).values
    y_test = df_test[args.target].values

    # Train with guarded fallback
    try:
        model = train_model(X_train, y_train, seed=seed, max_iter=args.max_iter)
        preds = model.predict(X_test)
        model_used = "MLPRegressor"
    except Exception:
        model = fallback_linear(X_train, y_train)
        preds = model.predict(X_test)
        model_used = "LinearRegression_fallback"

    metrics = compute_metrics(y_test, preds)

    # Write preds_model.json
    preds_out = [{"index": int(i), "prediction": float(p)} for i, p in enumerate(preds)]
    preds_path = outdir / "preds_model.json"
    write_json_atomic(preds_out, str(preds_path))

    # Write promotion_gate_summary.json
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_used": model_used,
        "seed": seed,
        "metrics": metrics,
        "n_features": int(X_test.shape[1]),
        "created_at": now_iso(),
        "script": {"path": str(Path(args.script_path).name), "sha256": None}
    }
    summary_path = outdir / "promotion_gate_summary.json"
    write_json_atomic(summary, str(summary_path))

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
        "scripts": {"canonical": {"path": str(Path(args.script_path).name), "sha256": script_sha}},
        "outputs": {
            "preds_model": {"path": str(preds_path), "sha256": preds_sha},
            "promotion_gate": {"path": str(summary_path), "sha256": summary_sha}
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
                   .format(run_id=run_id, script=str(script_path.name), sha=summary_sha, ts=now_iso()))
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
    p = argparse.ArgumentParser(description="train_phase2b2_HO.1A.py - canonical train pipeline")
    p.add_argument("--mode", choices=["synthetic", "random", "live"], default="synthetic")
    p.add_argument("--run-id", dest="run_id", default="synthetic_forecast_test_20251117")
    p.add_argument("--seed", type=int, default=20251117)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--n-features", dest="n_features", type=int, default=10)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--train-csv", dest="train_csv", default=None)
    p.add_argument("--test-csv", dest="test_csv", default=None)
    p.add_argument("--target", dest="target", default="target")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=500)
    p.add_argument("--script-path", dest="script_path", default=__file__)
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args)
