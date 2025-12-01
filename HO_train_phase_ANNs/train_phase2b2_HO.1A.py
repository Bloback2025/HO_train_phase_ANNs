#!/usr/bin/env python3
"""
train_phase2b2_HO.1B.py
Canonical, auditable minimal runner for phase2b2 HO.

- Deterministic synthetic-data path for smoke and audit runs.
- Produces: preds_model.json (list), preds_model.json.sha256.txt, promotion_gate_summary.json, run_manifest.<run_id>.patched.json, HANDOVER.CLOSURE.txt
- Writes files atomically and produces per-file SHA sidecars for audit.
- Minimal dependencies: numpy, scikit-learn, pandas (only for synthetic convenience).

Usage (example):
  python train_phase2b2_HO.1B.py --mode synthetic --run-id run_20251118_121954 --seed 20251117 --outdir ./ho_artifact_outputs

Notes:
- Deterministic by seed; suitable for reproducible smoke runs and handover.
- Safe-by-default: no network or external state writes outside outdir and captured_runs.
"""

from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd

VERSION = "1.0.0-canonical"

def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def write_json_atomic(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)

def make_synthetic(seed: int = 20251117, n_samples: int = 100, n_features: int = 4, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X.dot(coef) + noise * rng.randn(n_samples)
    cols = {f"f{i}": X[:, i] for i in range(n_features)}
    cols["target"] = y
    return pd.DataFrame(cols)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    residuals = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    r2 = float(1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true))**2)) if len(y_true) > 1 else 0.0
    return {"r2": r2, "mae": mae, "rmse": rmse, "n_test": int(len(y_true))}

def build_run_manifest(run_id: str, args: argparse.Namespace, preds_path: str, preds_sha: str, input_meta: Dict[str,Any]) -> Dict[str,Any]:
    return {
        "run_id": run_id,
        "version": VERSION,
        "timestamp_utc": now_iso(),
        "input_metadata": input_meta,
        "preds_path": preds_path,
        "preds_sha256": preds_sha,
        "seed": int(args.seed),
        "notes": "Deterministic synthetic run for handover; replace training block for production."
    }

def parse_args(argv: List[str]|None=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="train_phase2b2_HO.1B.py - minimal deterministic runner")
    p.add_argument("--mode", choices=["synthetic"], default="synthetic")
    p.add_argument("--run-id", dest="run_id", default="run_20251118_121954")
    p.add_argument("--seed", type=int, default=20251117)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-features", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--outdir", default="./ho_artifact_outputs")
    p.add_argument("--input-manifest", default=None, help="optional input manifest JSON with metadata and data (data optional)")
    p.add_argument("--script-path", dest="script_path", default=__file__)
    return p.parse_args(argv)

def run(args: argparse.Namespace) -> int:
    run_id = args.run_id
    outdir = Path(args.outdir) / run_id
    ensure_dir(outdir)

    # Load or create input manifest metadata
    input_manifest = {}
    if args.input_manifest:
        try:
            with open(args.input_manifest, "r", encoding="utf8") as f:
                input_manifest = json.load(f)
        except Exception as e:
            print(json.dumps({"status":"error","message":"failed to load input manifest","error": str(e)}))
            return 2
    else:
        input_manifest = {"metadata": {"source":"synthetic", "notes":"auto-generated for smoke run"}, "data": None}

    # Data path
    if args.mode == "synthetic":
        df = make_synthetic(seed=args.seed, n_samples=args.n_samples, n_features=args.n_features, noise=args.noise)
        df_train = df.sample(frac=0.7, random_state=int(args.seed))
        df_test = df.drop(df_train.index)
    else:
        print(json.dumps({"status":"error","message":"unsupported mode"}))
        return 3

    X_train = df_train.drop(columns=["target"]).values
    y_train = df_train["target"].values
    X_test = df_test.drop(columns=["target"]).values
    y_test = df_test["target"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Deterministic model: LinearRegression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    model_used = "LinearRegression"

    # Write preds as JSON array (atomic), and write SHA sidecar
    preds_path = str(outdir / "preds_model.json")
    preds_obj = {"run_id": run_id, "preds": [float(x) for x in preds], "seed": int(args.seed), "version": VERSION, "generated_at_utc": now_iso()}
    write_json_atomic(preds_obj, preds_path)
    preds_sha = sha256_of_file(preds_path)
    with open(preds_path + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(preds_sha)

    # Summary / promotion gate
    metrics = compute_metrics(y_test, preds)
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "created_at": now_iso(),
        "metrics": metrics,
        "n_features": int(X_test.shape[1]),
        "model_used": model_used,
        "script": {"path": str(Path(args.script_path).name), "sha256": None}
    }
    summary_path = str(outdir / "promotion_gate_summary.json")
    write_json_atomic(summary, summary_path)
    with open(summary_path + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(sha256_of_file(summary_path))

    # Manifest patch
    manifest_path = str(outdir / f"run_manifest.{run_id}.patched.json")
    manifest = build_run_manifest(run_id, args, preds_path, preds_sha, input_manifest.get("metadata", {}))
    write_json_atomic(manifest, manifest_path)

    # Closure file
    closure = Path("captured_runs") / run_id / "HANDOVER.CLOSURE.txt"
    ensure_dir(closure.parent)
    with open(str(closure), "w", encoding="utf8") as f:
        f.write(f"RUN_CLOSURE {now_iso()} script={Path(args.script_path).name}\n")

    # Emit compact audit JSON to stdout
    audit_line = {"status":"ok", "run_id": run_id, "preds_sha256": preds_sha, "preds_path": preds_path}
    print(json.dumps(audit_line, sort_keys=True))

    return 0

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    raise SystemExit(run(args))

# staged-marker
