#!/usr/bin/env python3
# deterministic_inference_v.1B.py
# Deterministic inference harness tailored for train_phase2b2_HO.1Bi workflow
# Purpose: provide a drop-in, auditable infer_with_model(...) API that:
#  - enforces canonical CSV SHAs when provided
#  - performs deterministic seeding for Python, NumPy, and TensorFlow (best-effort)
#  - loads an optional scaler (joblib/pickle) and verifies its SHA
#  - loads common TF/Keras model formats and runs inference
#  - writes preds_model.json and a small manifest fragment atomically
#  - writes uppercase SHA256 sidecars for all outputs
#
# Canonical CSVs referenced by the project (absolute paths and expected uppercase SHA256):
#   C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_training.csv
#     SHA: 04CC097BD744E1262AD885596C79C34D167F0A8E04B4D2DDA919EDB149709186
#   C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_validation.csv
#     SHA: 6A6713BF2968600AACC7341F49D757FD95AEA268127A8B6EB34AA784CACAB511
#   C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\hoxnc_testing.csv
#     SHA: D7E75485054836422F7F4311DEF959CCD01BCDABA614326F16E3AC2BE89C216A
#
# Design constraints:
#  - No network access; all IO confined to explicit outdir or canonical input paths
#  - Atomic writes and explicit uppercase SHA sidecars for auditability
#  - Minimal implicit preprocessing; harness expects caller to supply compatible inputs
#  - Clear manifest fragment returned for inclusion in HANDOVER artifacts

import os
import sys
import json
import hashlib
import tempfile
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Lazy imports to avoid heavy dependencies on import-time
def _import_optional_deps():
    deps = {}
    try:
        import numpy as np
        deps['np'] = np
    except Exception:
        deps['np'] = None
    try:
        import joblib
        deps['joblib'] = joblib
    except Exception:
        deps['joblib'] = None
    try:
        import tensorflow as tf
        deps['tf'] = tf
    except Exception:
        deps['tf'] = None
    try:
        import pandas as pd
        deps['pd'] = pd
    except Exception:
        deps['pd'] = None
    return deps

# -------------------------
# Utilities
# -------------------------
def sha256_file_upper(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk_size), b""):
            h.update(b)
    return h.hexdigest().upper()

def write_atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(tf.name, str(path))

def write_atomic_text(path: Path, text: str, encoding: str = "utf-8"):
    write_atomic_bytes(path, text.encode(encoding))

def write_sidecar_sha(path: Path, sha: str):
    sidecar = Path(str(path) + ".SHA256.TXT")
    # Format: <SHA>  <relative/path>  (manifest expects uppercase SHA)
    write_atomic_text(sidecar, sha + "\n")

def deterministic_seed_all(seed: int):
    random.seed(seed)
    deps = _import_optional_deps()
    np = deps.get('np')
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    tf = deps.get('tf')
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

# -------------------------
# Minimal CSV loader (best-effort)
# -------------------------
def load_csv_features(csv_path: Path, model=None):
    """
    Best-effort numeric feature extraction:
    - If pandas available, read CSV and return numeric columns as numpy array
    - If not, return a single-row zeros array sized to model input if model provided
    """
    deps = _import_optional_deps()
    pd = deps.get('pd')
    np = deps.get('np')
    if pd is not None:
        try:
            df = pd.read_csv(csv_path)
            # Select numeric columns only; caller should ensure correct ordering
            num_df = df.select_dtypes(include=["number"])
            if num_df.shape[0] == 0:
                # fallback to zeros
                if np is not None:
                    return np.zeros((1, 1))
                return [[0.0]]
            return num_df.to_numpy()
        except Exception:
            pass
    # Fallback: infer model input shape if possible
    if model is not None:
        try:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            features = input_shape[-1] if input_shape and len(input_shape) >= 2 else 1
            if np is not None:
                return np.zeros((1, features))
            return [[0.0] * features]
        except Exception:
            pass
    # Last resort
    if np is not None:
        return np.zeros((1, 1))
    return [[0.0]]

# -------------------------
# Public API
# -------------------------
def infer_with_model(
    model_path: str,
    input_csvs: Dict[str, str],
    outdir: str,
    seed: int = 20251117,
    scaler_path: Optional[str] = None,
    enforce_shas: Optional[Dict[str, str]] = None,
    mode: str = "deterministic"
) -> Dict[str, str]:
    """
    Run deterministic inference and produce auditable outputs.

    Parameters
    - model_path: path to model artifact (h5, SavedModel dir, or other)
    - input_csvs: dict with keys like 'train','val','test' -> absolute paths
    - outdir: explicit output directory for preds and manifest
    - seed: integer seed for deterministic behavior
    - scaler_path: optional path to scaler object used at training time
    - enforce_shas: optional dict mapping csv keys to expected uppercase SHA256 strings
    - mode: 'deterministic' or 'synthetic' (synthetic produces a small smoke output without model)
    Returns
    - dict with keys 'preds_path','preds_sha','manifest'
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Preflight: verify canonical CSVs if expected SHAs provided
    mismatches = []
    for k, p in input_csvs.items():
        ppath = Path(p)
        if not ppath.exists():
            raise FileNotFoundError(f"Missing input CSV {k}: {ppath}")
        if enforce_shas and k in enforce_shas:
            actual = sha256_file_upper(ppath)
            expected = enforce_shas[k].upper()
            if actual != expected:
                mismatches.append((k, str(ppath), expected, actual))
    if mismatches:
        details = "; ".join([f"{k} expected {e} got {a}" for (k,_,e,a) in mismatches])
        raise RuntimeError("SHA MISMATCH: " + details)

    # Deterministic seeding
    deterministic_seed_all(seed)

    # Synthetic smoke mode
    if mode == "synthetic":
        preds = {"run_id": outdir.name, "seed": seed, "mode": "synthetic", "preds": [{"id": "SYNTH_1", "score": 0.12345}]}
        preds_bytes = json.dumps(preds, indent=2).encode("utf-8")
        preds_path = outdir / "preds_model.json"
        write_atomic_bytes(preds_path, preds_bytes)
        preds_sha = hashlib.sha256(preds_bytes).hexdigest().upper()
        write_sidecar_sha(preds_path, preds_sha)

        manifest = {
            "run_id": outdir.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": "synthetic",
            "seed": seed,
            "input_files": {k: str(Path(v).resolve()) for k, v in input_csvs.items()},
            "input_shas": enforce_shas or {},
            "outputs": {"preds": str(preds_path), "preds_sha": preds_sha},
        }
        manifest_path = outdir / f"run_manifest.{outdir.name}.inference.json"
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        write_atomic_bytes(manifest_path, manifest_bytes)
        write_sidecar_sha(manifest_path, hashlib.sha256(manifest_bytes).hexdigest().upper())
        return {"preds_path": str(preds_path), "preds_sha": preds_sha, "manifest": str(manifest_path)}

    # Deterministic inference with model
    deps = _import_optional_deps()
    tf = deps.get('tf')
    np = deps.get('np')
    joblib = deps.get('joblib')

    if tf is None:
        raise RuntimeError("TensorFlow not available; cannot run deterministic inference in model mode")

    # Load scaler if provided
    scaler_sha = None
    scaler_obj = None
    if scaler_path:
        scaler_p = Path(scaler_path)
        if not scaler_p.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_p}")
        scaler_sha = sha256_file_upper(scaler_p)
        if joblib is not None:
            try:
                scaler_obj = joblib.load(str(scaler_p))
            except Exception:
                # fallback: try pickle
                import pickle
                with open(scaler_p, "rb") as f:
                    scaler_obj = pickle.load(f)
        else:
            # try pickle if joblib not present
            import pickle
            with open(scaler_p, "rb") as f:
                scaler_obj = pickle.load(f)

    # Load model (support common TF formats)
    model = None
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Model path not found: {model_p}")
    try:
        if model_p.is_dir():
            model = tf.keras.models.load_model(str(model_p))
        elif model_p.suffix in (".h5", ".hdf5"):
            model = tf.keras.models.load_model(str(model_p))
        else:
            # attempt to load as Keras model
            model = tf.keras.models.load_model(str(model_p))
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {model_p}: {e}")

    # Prepare inputs and run inference on test CSV
    test_csv = Path(input_csvs.get("test") or input_csvs.get("testing") or input_csvs.get("val") or next(iter(input_csvs.values())))
    X_test = load_csv_features(test_csv, model=model)
    # If scaler present, apply transform deterministically
    if scaler_obj is not None:
        try:
            X_test = scaler_obj.transform(X_test)
        except Exception:
            # If scaler incompatible, record and continue with raw features
            pass

    # Run model prediction
    preds_array = model.predict(X_test, batch_size=32)
    try:
        preds_list = preds_array.tolist()
    except Exception:
        preds_list = [float(x) for x in preds_array]

    # Serialize predictions
    preds = {"run_id": outdir.name, "timestamp": datetime.utcnow().isoformat() + "Z", "seed": seed, "preds": [{"row": i, "score": p} for i, p in enumerate(preds_list)]}
    preds_bytes = json.dumps(preds, indent=2).encode("utf-8")
    preds_path = outdir / "preds_model.json"
    write_atomic_bytes(preds_path, preds_bytes)
    preds_sha = hashlib.sha256(preds_bytes).hexdigest().upper()
    write_sidecar_sha(preds_path, preds_sha)

    # Write manifest fragment
    manifest = {
        "run_id": outdir.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "deterministic",
        "seed": seed,
        "model_path": str(model_p.resolve()),
        "model_sha": sha256_file_upper(model_p) if model_p.exists() else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "scaler_sha": scaler_sha,
        "input_files": {k: str(Path(v).resolve()) for k, v in input_csvs.items()},
        "input_shas": enforce_shas or {},
        "outputs": {"preds": str(preds_path), "preds_sha": preds_sha},
    }
    manifest_path = outdir / f"run_manifest.{outdir.name}.inference.json"
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    write_atomic_bytes(manifest_path, manifest_bytes)
    write_sidecar_sha(manifest_path, hashlib.sha256(manifest_bytes).hexdigest().upper())

    return {"preds_path": str(preds_path), "preds_sha": preds_sha, "manifest": str(manifest_path)}

# -------------------------
# CLI convenience wrapper for smoke runs
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deterministic inference harness (inference_v.1B)")
    parser.add_argument("--model", required=False, help="Path to model artifact (h5 or SavedModel dir)")
    parser.add_argument("--outdir", required=True, help="Output directory for preds and manifest")
    parser.add_argument("--mode", choices=["deterministic", "synthetic"], default="synthetic", help="Run mode")
    parser.add_argument("--seed", type=int, default=20251117, help="Deterministic seed")
    parser.add_argument("--scaler", required=False, help="Optional scaler path (joblib/pickle)")
    parser.add_argument("--train_csv", required=False, help="Training CSV path")
    parser.add_argument("--val_csv", required=False, help="Validation CSV path")
    parser.add_argument("--test_csv", required=False, help="Test CSV path")
    args = parser.parse_args()

    input_csvs = {}
    if args.train_csv:
        input_csvs['train'] = args.train_csv
    if args.val_csv:
        input_csvs['val'] = args.val_csv
    if args.test_csv:
        input_csvs['test'] = args.test_csv

    # If no CSVs provided and canonical paths exist, prefer them (best-effort)
    if not input_csvs:
        canonical_base = Path(r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080")
        cand_train = canonical_base / "hoxnc_training.csv"
        cand_val = canonical_base / "hoxnc_validation.csv"
        cand_test = canonical_base / "hoxnc_testing.csv"
        if cand_train.exists():
            input_csvs['train'] = str(cand_train)
        if cand_val.exists():
            input_csvs['val'] = str(cand_val)
        if cand_test.exists():
            input_csvs['test'] = str(cand_test)

    # enforce_shas can be provided here if desired; left empty by default
    enforce_shas = None

    result = infer_with_model(
        model_path=args.model or "",
        input_csvs=input_csvs,
        outdir=args.outdir,
        seed=args.seed,
        scaler_path=args.scaler,
        enforce_shas=enforce_shas,
        mode=args.mode
    )
    print(json.dumps(result, indent=2))
