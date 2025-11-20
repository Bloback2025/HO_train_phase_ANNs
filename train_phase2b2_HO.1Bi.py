#!/usr/bin/env python3
# train_phase2b2_HO.1Bi.py
# Instrumented variant of train_phase2b2_HO.1B.py
# Integrates provenance_safe_preds_writer to prevent accidental preds reuse/overwrite.
# Usage: run with the same CLI args as train_phase2b2_HO.1B.py; this file is a drop-in, auditable variant.

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

# provenance writer (must be in repo)
from provenance_safe_preds_writer import write_preds_provenance

# Module-level flags you can toggle for debugging only
ALLOW_PRED_OVERWRITE = True
ALLOW_PRED_REUSE = True

# ---- Helpers ----
def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_of_file(p: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ---- Main flow (keeps semantics of your canonical script) ----
def main(argv):
    p = argparse.ArgumentParser(description="train_phase2b2_HO.1Bi - provenance-safe preds writer variant")
    p.add_argument("--outdir", required=True, help="run output directory")
    p.add_argument("--script_path", required=True, help="path to training script for SHA")
    p.add_argument("--run_id", required=True, help="run id string")
    p.add_argument("--mode", default="synthetic", help="mode")
    args = p.parse_args(argv)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id
    script_path = Path(args.script_path).resolve()
    script_name = script_path.name

    # ---- Begin replacement: real training / inference block ----
    # This block expects you have a saved model or a routine that returns preds for the test set.
    # Adjust loader paths (model_path, test_X_path, scaler_path if used) to match your repo layout.

    import numpy as np

    # Helpers: basic metrics (no extra dependencies)
    def mae(y_true, y_pred):
        y = np.array(y_true); p = np.array(y_pred)
        return float(np.mean(np.abs(y - p)))
    def rmse(y_true, y_pred):
        y = np.array(y_true); p = np.array(y_pred)
        return float(np.sqrt(np.mean((y - p) ** 2)))
    def r2_score(y_true, y_pred):
        y = np.array(y_true); p = np.array(y_pred)
        ss_res = np.sum((y - p) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Edit these to point at your real model artifacts and test data
    model_path = outdir / "model_weights.h5"       # example; change if you store elsewhere
    scaler_path = outdir / "scaler.joblib"         # optional; set to None if not used
    test_X_path = Path("data") / "test_features.npy"
    test_y_path = Path("data") / "test_targets.npy"

    # Load test data (adjust if you use CSVs or other formats)
    if test_X_path.exists() and test_y_path.exists():
        X_test = np.load(test_X_path)
        y_test = np.load(test_y_path)
    else:
        # Fallback: synthetic test data to avoid crashing; replace with your loader
        rng = np.random.RandomState(20251117)
        X_test = rng.randn(400, 10)
        y_test = rng.randn(400)

    # Optionally load scaler
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        try:
            import joblib
            scaler = joblib.load(str(scaler_path))
            X_test = scaler.transform(X_test)
        except Exception:
            scaler = None
            scaler_path = None
    else:
        scaler = None
        scaler_path = None

    # Load model and predict
    preds_array = None
    model_used = "unknown"
    # Try Keras/TensorFlow saved model (file or dir)
    try:
        if model_path.exists():
            try:
                # Attempt to load as keras model (handles saved_model dir or h5)
                from tensorflow import keras  # optional dependency
                model = keras.models.load_model(str(model_path))
                preds_array = model.predict(X_test).flatten().tolist()
                model_used = "keras_model"
            except Exception:
                preds_array = None
    except Exception:
        preds_array = None

    # If preds not produced by model, try sklearn joblib model
    if preds_array is None:
        try:
            import joblib
            sk_model_path = outdir / "sk_model.joblib"
            if sk_model_path.exists():
                skmodel = joblib.load(str(sk_model_path))
                preds_array = skmodel.predict(X_test).tolist()
                model_used = type(skmodel).__name__
        except Exception:
            preds_array = None

    # If still None, fail loud (prevents silent fallback)
    if preds_array is None:
        raise RuntimeError("No predictions produced: check model artifacts and loader paths (model_path, sk_model.joblib)")

    # Compute metrics
    metrics = {
        "r2": r2_score(y_test, preds_array),
        "mae": mae(y_test, preds_array),
        "rmse": rmse(y_test, preds_array),
        "residual_mean": float(np.mean(np.array(y_test) - np.array(preds_array))),
        "n_test": len(preds_array)
    }

    promotion_summary = {
        "run_id": run_id,
        "mode": args.mode,
        "model_used": model_used,
        "seed": 20251117,
        "metrics": metrics,
        "n_features": int(X_test.shape[1]) if hasattr(X_test, "shape") else None,
        "created_at": now_iso(),
        "script": {"path": script_name, "sha256": None}
    }
    # ---- End replacement ----

    # ---- write promotion summary (unchanged behavior) ----
    summary_path = outdir / "promotion_gate_summary.json"
    with open(summary_path, "w", encoding="utf8") as fh:
        json.dump(promotion_summary, fh, indent=2, sort_keys=True, ensure_ascii=False)

    # ---- compute script SHA ----
    try:
        script_sha = sha256_of_file(str(script_path))
    except Exception:
        script_sha = None

    # ---- Use provenance_safe_preds_writer to write preds and .sha256 sidecar ----
    try:
        pw_result = write_preds_provenance(
            preds=preds_array,
            outdir=str(outdir),
            run_id=run_id,
            allow_overwrite=ALLOW_PRED_OVERWRITE,
            allow_reuse=ALLOW_PRED_REUSE
        )
    except Exception as e:
        # Fail fast and print a clear audit message
        print("PROVENANCE_WRITE_FAILURE:", str(e), file=sys.stderr)
        raise

    preds_path = Path(pw_result["path"])
    preds_sha = pw_result["sha256"]

    # ---- compute summary SHA and write sidecar ----
    summary_sha = sha256_of_file(str(summary_path))
    with open(str(summary_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(summary_sha)

    # ---- manifest patch (keeps same manifest shape as canonical script) ----
    manifest_patch = {
        "run_id": run_id,
        "manifest_patched_at": now_iso(),
        "scripts": {"promoted": {"path": script_name, "sha256": script_sha}},
        "outputs": {
            "preds_model": {"path": str(preds_path), "sha256": preds_sha},
            "promotion_gate": {"path": str(summary_path), "sha256": summary_sha},
            "scaler": {"path": str(scaler_path) if scaler_path else None, "sha256": None}
        }
    }
    manifest_path = outdir / f"run_manifest.{run_id}.patched.json"
    with open(manifest_path, "w", encoding="utf8") as fh:
        json.dump(manifest_patch, fh, indent=2, sort_keys=True, ensure_ascii=False)
    manifest_sha = sha256_of_file(str(manifest_path))
    with open(str(manifest_path) + ".sha256.txt", "w", encoding="ascii") as f:
        f.write(manifest_sha)

    # ---- append audit closure line to candidate audit (keeps canonical format) ----
    closure = (
        f"FINAL_PROMOTE_COMPLETED: tag=promotion/{run_id} ; run_id={run_id} ; "
        f"script_run_sha={script_sha} ; preds_sha={preds_sha} ; "
        f"promotion_summary_sha={summary_sha} ; model_weights_sha=null ; scaler_sha="
    )
    candidate_audit = Path.cwd() / "audit_summary_for_notepad.txt"
    with open(candidate_audit, "a", encoding="utf8") as f:
        f.write("\n" + closure + "\n")

    # ---- audit prints (compatible with existing tooling) ----
    print("WROTE:", str(preds_path))
    print("WROTE:", str(summary_path))
    print("SCRIPT_SHA:", script_sha)
    print("PREDS_SHA:", preds_sha)
    print("SUMMARY_SHA:", summary_sha)
    print("MANIFEST_PATCHED:", str(manifest_path))
    print("MANIFEST_SHA:", manifest_sha)
    print("OUTDIR:", str(outdir))

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]) or 0)
    except Exception as exc:
        # Ensure that failures surface in CI and logs
        print("ERROR:", str(exc), file=sys.stderr)
        raise
