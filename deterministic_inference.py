#!/usr/bin/env python3
"""
deterministic_inference.py
Safe, idempotent inference entrypoint for OHLC model.

Usage (example):
    python deterministic_inference.py --manifest hoxnc_training.with_base_and_lags.manifest.json --csv hoxnc_full.csv --model models/ohlc_best.keras

Key behaviors:
- Creates Close_t+1 from Close if missing.
- Derives features from manifest or CSV header (conservative).
- Selective dropna on required columns only.
- Loads models/scaler.pkl if present; otherwise fits a fallback scaler on the training slice.
- Deterministic chronological split and metrics printed as [METRICS].
- Prints RUN_INFO / RUN_WARN / RUN_ERROR and final RUN_COMPLETE markers for audit.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Constants for splits (mirror training script proportions)
TEST_FRAC = 0.15
VAL_FRAC = 0.10  # fraction of remaining after removing test as in previous pipeline


def eprint(*args, **kwargs):
    """Audit-friendly stderr prints."""
    print(*args, file=sys.stderr, **kwargs)


def load_manifest_features(manifest_path: Optional[Path]) -> Optional[List[str]]:
    if not manifest_path:
        return None
    try:
        with manifest_path.open("r", encoding="utf8") as fh:
            m = json.load(fh)
        f = m.get("features")
        if isinstance(f, list) and f:
            return [str(x) for x in f]
    except Exception as e:
        eprint("RUN_WARN: manifest_load_failed", manifest_path, str(e))
    return None


def conservative_feature_fallback(csv_path: Path, target_name: str, max_lag: int = 0) -> List[str]:
    try:
        head = pd.read_csv(csv_path, nrows=1)
        cols = list(head.columns)
    except Exception:
        cols = []
    features = [c for c in cols if c not in ("Date", target_name)]
    if features:
        return features
    features = []
    for base in ["Open", "High", "Low", "Close"]:
        for k in range(1, max(1, max_lag) + 1):
            features.append(f"{base}_lag{k}")
    if not features and "Close" in cols:
        return ["Close"]
    return features


def chronological_split(
    X: np.ndarray, y: np.ndarray, test_frac: float = TEST_FRAC, val_frac: float = VAL_FRAC
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    test_n = int(round(test_frac * n))
    val_n = int(round(val_frac * (n - test_n)))
    train_end = n - test_n - val_n
    val_end = train_end + val_n
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def safe_cast_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            try:
                df[c] = df[c].astype(float)
            except Exception:
                df[c] = df[c]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None, help="Manifest JSON path (optional)")
    parser.add_argument("--csv", type=str, default="hoxnc_full.csv", help="Input CSV for inference")
    parser.add_argument("--model", type=str, required=True, help="Keras model directory or file")
    parser.add_argument("--target", type=str, default="Close_t+1", help="Target column name to use/create")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC, help="Test fraction")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC, help="Val fraction of remaining")
    args = parser.parse_args()

    try:
        manifest_path = Path(args.manifest) if args.manifest else None
        csv_path = Path(args.csv)
        model_path = Path(args.model)
        target_col = args.target

        if not csv_path.exists():
            eprint("RUN_ERROR: csv_not_found", str(csv_path))
            print("RUN_COMPLETE: FAILURE")
            return 1

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            eprint("RUN_ERROR: csv_read_failed", str(e))
            print("RUN_COMPLETE: FAILURE")
            return 1

        features = load_manifest_features(manifest_path)

        max_lag = 0
        if features:
            for f in features:
                if "_lag" in f:
                    try:
                        max_lag = max(max_lag, int(f.split("_lag")[-1]))
                    except Exception:
                        pass
        else:
            cols = list(df.columns)
            for c in cols:
                if "_lag" in c:
                    try:
                        max_lag = max(max_lag, int(c.split("_lag")[-1]))
                    except Exception:
                        pass

        if not features:
            features = conservative_feature_fallback(csv_path, target_col, max_lag)

        if (target_col not in df.columns) and ("Close" in df.columns):
            try:
                df[target_col] = df["Close"].shift(-1)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                eprint("RUN_INFO: created_target", target_col, "rows=", len(df))
            except Exception as e:
                eprint("RUN_WARN: target_creation_failed", str(e))

        _required_cols = list(features) + ([target_col] if (target_col in df.columns) else [])

        try:
            df = df.dropna(axis=0, subset=_required_cols).reset_index(drop=True)
        except Exception as e:
            eprint("RUN_ERROR: selective_dropna_failed", str(e))
            print("RUN_COMPLETE: FAILURE")
            return 1

        df = safe_cast_float(df, [c for c in _required_cols if c in df.columns])

        missing = [c for c in features if c not in df.columns]
        if missing:
            eprint("RUN_ERROR: missing_after_preprocessing", missing)
            print("RUN_COMPLETE: FAILURE")
            return 1

        X = df[features].astype(float).values
        y = df[target_col].astype(float).values

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X, y, test_frac=args.test_frac, val_frac=args.val_frac)

        scaler = None
        scaler_path = Path("models") / "scaler.pkl"
        if scaler_path.exists():
            try:
                with scaler_path.open("rb") as fh:
                    scaler = pickle.load(fh)
                eprint("RUN_INFO: loaded_scaler", str(scaler_path))
            except Exception as e:
                eprint("RUN_WARN: failed_loading_scaler", str(e))

        if scaler is None:
            try:
                scaler = StandardScaler().fit(X_train)
                eprint("RUN_INFO: fitted_fallback_scaler_on_train")
            except Exception as e:
                eprint("RUN_WARN: scaler_fit_failed", str(e))

        X_test_s = scaler.transform(X_test) if scaler is not None else X_test

        try:
            model = tf.keras.models.load_model(str(model_path))
        except Exception as e:
            eprint("RUN_ERROR: model_load_failed", str(e))
            print("RUN_COMPLETE: FAILURE")
            return 1

        try:
            y_pred = model.predict(X_test_s, verbose=0).reshape(-1)
        except Exception as e:
            eprint("RUN_ERROR: model_predict_failed", str(e))
            print("RUN_COMPLETE: FAILURE")
            return 1

        try:
            mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else float("nan")
            r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 0 else float("nan")
            print(f"[METRICS] test_mae={mae:.6f} r2_ann={r2:.6f}")
        except Exception as e:
            eprint("RUN_WARN: metrics_compute_failed", str(e))
            print("[METRICS] test_mae=nan r2_ann=nan")

        print("RUN_COMPLETE: SUCCESS")
        return 0

    except Exception as exc:
        eprint("RUN_ERROR: unhandled_exception", str(exc))
        print("RUN_COMPLETE: FAILURE")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
