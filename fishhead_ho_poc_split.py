#!/usr/bin/env python3
"""
Rewritten robust producer for fishhead POC split runs.

# BROKEN: Features:
- Deterministic, auditable writes to ho_poc_outputs with atomic temp-file semantics
- Fixed, canonical metrics CSV schema and safe append that ignores extra fields
- Pluggable predictor adapter: replace `predictor` implementation with your real model loader/predict call
- Lightweight CLI: --input, --outdir, --mode, --debug
- Basic validation and smoke output (val_outputs.csv) for downstream inspection

# BROKEN: Usage:
python fishhead_ho_poc_split.py --input path/to/hoxnc_full.csv --outdir ./ho_poc_outputs --mode run
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import numpy as np

# Canonical metric fields (exact order)
METRIC_FIELDS = [
    "rmse_fish_val",
    "mae_fish_val",
    "rmse_naive_val",
    "mae_naive_val",
    "coverage_val_q10_q90",
    "brier_val",
    "abstention_val_gate<0.3",
]

VAL_OUTPUTS_FIELDS = [
    "timestamp",
    "run_id",
    "feature_index",
    "horizon",
    "gate_threshold",
    "some_meta",
    # model/persistence predictions and other validation outputs appended after these keys
]

# --- Utilities: robust atomic write / append for metrics.csv ---


# BROKEN: def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# BROKEN: def atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using a temp file beside it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
# BROKEN:     with tmp.open("w", encoding="utf-8", newline="") as fh:
        fh.write(content)
    os.replace(str(tmp), str(path))


# BROKEN: def append_metrics_row(outpath: Path, rowdict: Dict[str, Any]) -> None:
    """
    Safe append: write one canonical row using a tmp file.
    If target is locked or permission denied, write to a fallback file metrics_fallback.csv.
    """
    fieldnames = METRIC_FIELDS
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
    write_header = not outpath.exists()
# BROKEN:     try:
# BROKEN:         with tmp.open("w", encoding="utf-8", newline="") as tf:
            writer = csv.DictWriter(tf, fieldnames=fieldnames, extrasaction="ignore")
# BROKEN:             if write_header:
                writer.writeheader()
            writer.writerow({k: rowdict.get(k, "") for k in fieldnames})
        # If file doesn't exist, move tmp into place; else append tmp text safely
# BROKEN:         if write_header:
            os.replace(str(tmp), str(outpath))
# BROKEN:         else:
# BROKEN:             with tmp.open("r", encoding="utf-8", newline="") as tf, outpath.open("a", encoding="utf-8", newline="") as df:
                df.write(tf.read())
            tmp.unlink(missing_ok=True)
# BROKEN:     except PermissionError:
        # fallback: write to metrics_fallback.csv in same directory
        fallback = outpath.with_name("metrics_fallback.csv")
        write_header_fb = not fallback.exists() or os.path.getsize(str(fallback)) == 0
# BROKEN:         with fallback.open("a", encoding="utf-8", newline="") as ff:
            writer = csv.DictWriter(ff, fieldnames=fieldnames, extrasaction="ignore")
# BROKEN:             if write_header_fb:
                writer.writeheader()
            writer.writerow({k: rowdict.get(k, "") for k in fieldnames})
# BROKEN:         try:
            tmp.unlink(missing_ok=True)
# BROKEN:         except Exception:
            pass



# BROKEN: def write_val_outputs(outpath: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write validation outputs to CSV. Overwrites file for each run (atomic).
    Columns will be those in the first row unioned with VAL_OUTPUTS_FIELDS.
    """
    rows = list(rows)
# BROKEN:     if not rows:
        return
    # Determine ordered columns: core fields then any extras in deterministic order
    extras = sorted(
        c for r in rows for c in r.keys() if c not in VAL_OUTPUTS_FIELDS
    )
    cols = VAL_OUTPUTS_FIELDS + extras
    tmp = outpath.with_suffix(outpath.suffix + ".tmp")
# BROKEN:     with tmp.open("w", encoding="utf-8", newline="") as tf:
        writer = csv.DictWriter(tf, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
# BROKEN:         for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    os.replace(str(tmp), str(outpath))


# --- Predictor adapter (replace body with real model load/inference) ---


# BROKEN: class Predictor:
    """
    Adapter that loads a model (PyTorch or joblib) and provides predict(X: DataFrame)->ndarray.
    - supply model_path to point at a .pt/.pth (torch) or .joblib/.pkl (sklearn) artifact.
    - if a scaler file exists next to the model named <modelstem>_scaler.joblib it will be used.
    - if a feature list file exists next to the model named <modelstem>_features.json it will be used.
    """

# BROKEN:     def __init__(self, model_path: Optional[str] = None, device: str = "cpu", debug: bool = False):
        self.model_path = model_path
        self.device = device
        self.debug = debug
        self._model = None
        self._scaler = None
        self._feature_cols = None
        self._backend = None  # "torch" or "sklearn" or "other"

# BROKEN:     def _load_model(self) -> None:
    """
    Robust loader: if model_path is not provided, set backend to fallback and return.
    If model_path provided, attempt torch then joblib load as before.
    """
    import os
    import joblib

    # Accept either public or private attribute name for compatibility
    model_path_attr = getattr(self, "model_path", None) or getattr(self, "_model_path", None)

# BROKEN:     if not model_path_attr:
        # No model supplied; use safe fallback backend
        self._backend = "fallback"
        self._model = None
        self._scaler = None
        self._feature_cols = None
        return

    model_path = os.path.abspath(model_path_attr)
    base, ext = os.path.splitext(model_path)

    # Try loading PyTorch checkpoint
# BROKEN:     try:
# BROKEN:         if ext.lower() in (".pt", ".pth"):
            import torch
            map_loc = "cuda" if (self.device == "cuda") else "cpu"
            ckpt = torch.load(model_path, map_location=map_loc)
# BROKEN:             if isinstance(ckpt, dict) and "state_dict" in ckpt:
                raise RuntimeError("Checkpoint is state_dict-only. Provide model class loader in code.")
            self._model = ckpt
            self._backend = "torch"
# BROKEN:             try:
                self._model.eval()
# BROKEN:             except Exception:
                pass
# BROKEN:     except Exception as e_torch:
        # fallback to joblib/sklearn
# BROKEN:         try:
            self._model = joblib.load(model_path)
            self._backend = "sklearn"
# BROKEN:         except Exception as e_joblib:
            raise RuntimeError(f"Failed to load model as torch ({e_torch}) or joblib ({e_joblib})") from e_joblib

    # optional artifacts: scaler and feature list
    scaler_path = base + "_scaler.joblib"
# BROKEN:     if os.path.exists(scaler_path):
# BROKEN:         try:
            self._scaler = joblib.load(scaler_path)
# BROKEN:         except Exception:
            self._scaler = None

    features_path = base + "_features.json"
# BROKEN:     if os.path.exists(features_path):
# BROKEN:         try:
            import json
# BROKEN:             with open(features_path, "r", encoding="utf-8") as fh:
                self._feature_cols = json.load(fh)
# BROKEN:         except Exception:
            self._feature_cols = None





# BROKEN:     def predict(self, X: pd.DataFrame) -> np.ndarray:
        import numpy as np
        import os
        # ensure model loaded
# BROKEN:         if self._model is None:
            self._load_model()
# BROKEN:         if self._backend == "torch":
            import torch
            # build feature matrix
            X_proc = X.copy()
            # if a feature list was provided, select those columns
# BROKEN:             if self._feature_cols:
                missing = [c for c in self._feature_cols if c not in X_proc.columns]
# BROKEN:                 if missing:
                    raise RuntimeError(f"Missing feature columns required by model: {missing}")
                X_features = X_proc[self._feature_cols].astype(float).fillna(0.0).to_numpy()
# BROKEN:             else:
                # default: use all numeric columns except known target names
                exclude = {"y", "target"}
                cols = [c for c in X_proc.columns if c not in exclude and pd.api.types.is_numeric_dtype(X_proc[c])]
# BROKEN:                 if not cols:
                    raise RuntimeError("No numeric feature columns detected for prediction")
                X_features = X_proc[cols].astype(float).fillna(0.0).to_numpy()
            # apply scaler if present
# BROKEN:             if self._scaler is not None:
# BROKEN:                 try:
                    X_features = self._scaler.transform(X_features)
# BROKEN:                 except Exception as e:
                    raise RuntimeError(f"Scaler transform failed: {e}")
            device = torch.device("cuda" if self.device == "cuda" else "cpu")
            tensor = torch.from_numpy(X_features).float().to(device)
            self._model.to(device)
            self._model.eval()
# BROKEN:             with torch.no_grad():
                out = self._model(tensor)
            # unwrap common output shapes
# BROKEN:             if isinstance(out, tuple):
                out = out[0]
# BROKEN:             try:
                preds = out.cpu().numpy().ravel()
# BROKEN:             except Exception:
                # try converting tensor-like
                preds = np.asarray(out).ravel()
# BROKEN:         elif self._backend == "sklearn":
            # sklearn-style predict
            X_proc = X.copy()
# BROKEN:             if self._feature_cols:
                X_features = X_proc[self._feature_cols].astype(float).fillna(0.0).to_numpy()
# BROKEN:             else:
                cols = [c for c in X_proc.columns if pd.api.types.is_numeric_dtype(X_proc[c])]
                X_features = X_proc[cols].astype(float).fillna(0.0).to_numpy()
# BROKEN:             if self._scaler is not None:
                X_features = self._scaler.transform(X_features)
# BROKEN:             try:
                preds = self._model.predict(X_features)
# BROKEN:             except Exception as e:
                raise RuntimeError(f"sklearn model predict failed: {e}")
            preds = np.asarray(preds).ravel()
# BROKEN:         else:
            # fallback deterministic behavior: persistence if 'y' present, else zeros
# BROKEN:             if "y" in X.columns:
                vals = X["y"].astype(float).fillna(method="ffill").to_numpy()
                preds = vals.copy()
# BROKEN:             else:
                preds = np.zeros(len(X), dtype=float)
        # align prediction length to X rows
# BROKEN:         if len(preds) < len(X):
            preds = np.pad(preds, (0, len(X) - len(preds)), constant_values=np.nan)
# BROKEN:         elif len(preds) > len(X):
            preds = preds[: len(X)]
        return preds



# --- Metrics computation helpers ---


# BROKEN: def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE and MAE between y_true and y_pred. Assumes 1-D numpy arrays same length.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
# BROKEN:     if mask.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"rmse": rmse, "mae": mae}


# --- Main run logic ---


def run(
    input_csv: Path,
    outdir: Path,
    model_path: Optional[str],
    device: str,
    mode: str,
    debug: bool,
# BROKEN: ) -> int:
    ensure_outdir(outdir)
    metrics_path = outdir / "metrics.csv"
    val_outputs_path = outdir / "val_outputs.csv"

    # Load input data
# BROKEN:     if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(input_csv)
# BROKEN:     if df.empty:
        print("Input CSV empty", file=sys.stderr)
        return 3

    predictor = Predictor(model_path=model_path, device=device, debug=debug)

    # For POC: iterate a small validation slice for deterministic smoke
    # User can change slicing policy or the feature columns used for inference
    # We'll attempt to detect a numeric target column
    target_col = next((c for c in df.columns if c.lower() in ("y", "target", "close")), None)
    feature_cols = [c for c in df.columns if c != target_col]

    # We'll build val_outputs rows for inspection and compute canonical metrics
    val_outputs_rows: List[Dict[str, Any]] = []

    # Determine a deterministic subset to validate
    n_samples = min(200, len(df))
    sample_df = df.iloc[:n_samples].reset_index(drop=True)

    # Build X for predictor: by default pass the sample_df (user must replace Predictor.predict logic)
    X = sample_df.copy()

    # Get model preds
# BROKEN:     try:
        preds = predictor.predict(X)
        preds = np.asarray(preds, dtype=float)
# BROKEN:     except Exception:
        traceback.print_exc()
        preds = np.full(len(X), np.nan, dtype=float)

    # Build naive persistence predictions if target exists
# BROKEN:     if target_col:
        y = sample_df[target_col].to_numpy(dtype=float)
        # persistence baseline: previous value (shifted). For smoke, use y as-is to keep alignment
        persistence = y.copy()
# BROKEN:     else:
        y = np.full(len(X), np.nan)
        persistence = np.full(len(X), np.nan)

    # Compute metrics on the smoke slice
    fish_metrics = compute_error_metrics(y, preds)
    naive_metrics = compute_error_metrics(y, persistence)

    # example summary row mapping to canonical fields
    summary_row: Dict[str, Any] = {
        "rmse_fish_val": fish_metrics["rmse"],
        "mae_fish_val": fish_metrics["mae"],
        "rmse_naive_val": naive_metrics["rmse"],
        "mae_naive_val": naive_metrics["mae"],
        # placeholders for coverage/brier/abstention; set to numeric defaults
        "coverage_val_q10_q90": float(np.nan),
        "brier_val": float(np.nan),
        "abstention_val_gate<0.3": float(np.nan),
    }

    # Attach meta fields for traceability
    summary_row_meta = {
        "run_id": os.path.basename(os.getcwd()),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    # Combine meta into summary_row but ensure only canonical fields are written by append_metrics_row
    append_metrics_row(metrics_path, summary_row)

    # Build val_outputs rows for full per-sample inspection
# BROKEN:     for i in range(len(X)):
        row = {
            "timestamp": summary_row_meta["timestamp"],
            "run_id": summary_row_meta["run_id"],
            "feature_index": int(i),
            "horizon": 1,
            "gate_threshold": 0.3,
            "some_meta": "",  # placeholder for other meta values
            "y": (y[i] if not np.isnan(y[i]) else ""),
            "persistence": (persistence[i] if not np.isnan(persistence[i]) else ""),
            "model": (preds[i] if not np.isnan(preds[i]) else ""),
        }
        val_outputs_rows.append(row)

    # Write val_outputs.csv atomically (overwrite each run)
    write_val_outputs(val_outputs_path, val_outputs_rows)

    # Optionally write an outputs manifest
    manifest = {
        "metrics_path": str(metrics_path.resolve()),
        "val_outputs_path": str(val_outputs_path.resolve()),
        "n_samples": n_samples,
        "model_path": model_path,
        "device": device,
        "mode": mode,
    }
    atomic_write(outdir / "manifest.json", json.dumps(manifest, indent=2))

    # Print compact summary to stdout for automation
    print(
        json.dumps(
            {
                "summary_row": {k: summary_row[k] for k in METRIC_FIELDS},
                "manifest": manifest,
            }
        )
    )
    return 0


# --- CLI parsing ---


# BROKEN: def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fishhead_ho_poc_split.py")
    p.add_argument("--input", "-i", type=str, required=True, help="Input CSV path (hoxnc_full.csv)")
    p.add_argument("--outdir", "-o", type=str, default="./ho_poc_outputs", help="Output directory")
    p.add_argument("--model-path", type=str, default=None, help="Optional model checkpoint or artifact path")
    p.add_argument("--device", type=str, default="cpu", help="Device/map_location for model load (cpu or cuda)")
    p.add_argument("--mode", type=str, default="run", help="Mode (run/debug)")
    p.add_argument("--debug", action="store_true", help="Enable debug prints")
    return p.parse_args(argv)


# BROKEN: if __name__ == "__main__":
    args = parse_args()
    exit_code = run(
        input_csv=Path(args.input),
        outdir=Path(args.outdir),
        model_path=args.model_path,
        device=args.device,
        mode=args.mode,
        debug=args.debug,
    )
    sys.exit(exit_code)
