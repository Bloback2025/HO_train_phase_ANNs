#!/usr/bin/env python3
"""
deterministic_inference.py

Usage example:
  python deterministic_inference.py \
    --train-csv "C:\... \hoxnc_training.csv" \
    --val-csv   "C:\... \hoxnc_validation.csv" \
    --test-csv  "C:\... \hoxnc_testing.csv" \
    --model models/ohlc_best.keras \
    --deterministic-inference \
    --deterministic-seed 315
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hashlib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

TEST_FRAC = 0.15
VAL_FRAC = 0.10

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest_features(manifest_path: Optional[Path]) -> Optional[List[str]]:
    if not manifest_path:
        return None
    try:
        with Path(manifest_path).open("r", encoding="utf8") as fh:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    parser.add_argument("--csv", type=str, default="hoxnc_full.csv", help="Input CSV for inference (fallback)")
    parser.add_argument("--train-csv", dest="train_csv", type=str, default=None, help="Training CSV (canonical)")
    parser.add_argument("--val-csv",   dest="val_csv",   type=str, default=None, help="Validation CSV (canonical)")
    parser.add_argument("--test-csv",  dest="test_csv",  type=str, default=None, help="Test CSV (canonical)")
    parser.add_argument("--model", type=str, required=True, help="Keras model directory or file")
    parser.add_argument("--target", type=str, default="Close_t+1", help="Target column name to use/create")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC, help="Test fraction (combined-mode)")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC, help="Val fraction (combined-mode)")
    parser.add_argument("--deterministic-inference", action="store_true", help="Enable deterministic inference and log deterministic_meta")
    parser.add_argument("--deterministic-seed", type=int, default=315, help="Seed for deterministic inference (default 315)")
    args = parser.parse_args()

    try:
        manifest_path = Path(args.manifest) if args.manifest else None
        target_col = args.target
        model_path = Path(args.model)

        deterministic_meta = {"deterministic": False}
        if args.deterministic_inference:
            SEED = int(args.deterministic_seed)
            random.seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
            deterministic_meta = {
                "deterministic": True,
                "seed": SEED,
                "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS", "<unset>")
            }
        eprint("RUN_INFO: deterministic_meta", json.dumps(deterministic_meta))

        # Load CSV(s)
        if args.train_csv and args.test_csv:
            train_path = Path(args.train_csv)
            val_path = Path(args.val_csv) if args.val_csv else None
            test_path = Path(args.test_csv)

            for p in [train_path, test_path] + ([val_path] if val_path else []):
                if p and not p.exists():
                    eprint("RUN_ERROR: csv_not_found", str(p))
                    print("RUN_COMPLETE: FAILURE")
                    return 1

            eprint("RUN_INFO: dataset", json.dumps({
                "train": {"path": str(train_path.resolve()), "sha256": file_sha256(train_path)},
                "val": ({"path": str(val_path.resolve()), "sha256": file_sha256(val_path)} if val_path else None),
                "test": {"path": str(test_path.resolve()), "sha256": file_sha256(test_path)}
            }))

            try:
                df_train = pd.read_csv(train_path)
                df_val = pd.read_csv(val_path) if val_path else None
                df_test = pd.read_csv(test_path)
            except Exception as e:
                eprint("RUN_ERROR: csv_read_failed", str(e))
                print("RUN_COMPLETE: FAILURE")
                return 1

            df = pd.concat([d for d in [df_train, df_val, df_test] if d is not None], ignore_index=True)
        else:
            csv_path = Path(args.csv)
            if not csv_path.exists():
                eprint("RUN_ERROR: csv_not_found", str(csv_path))
                print("RUN_COMPLETE: FAILURE")
                return 1
            eprint("RUN_INFO: dataset", json.dumps({"csv": {"path": str(csv_path.resolve()), "sha256": file_sha256(csv_path)}}))
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                eprint("RUN_ERROR: csv_read_failed", str(e))
                print("RUN_COMPLETE: FAILURE")
                return 1
            df_train = df_val = df_test = None

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
            fallback_csv_path = Path(args.train_csv) if args.train_csv and Path(args.train_csv).exists() else (Path(args.csv) if Path(args.csv).exists() else None)
            if fallback_csv_path:
                features = conservative_feature_fallback(fallback_csv_path, target_col, max_lag)
            else:
                tmp_head = Path(".tmp_feature_head.csv")
                try:
                    df.head(1).to_csv(tmp_head, index=False)
                    features = conservative_feature_fallback(tmp_head, target_col, max_lag)
                finally:
                    if tmp_head.exists():
                        tmp_head.unlink()

        # Robust target creation and propagation
        def _propagate_target(df_frame: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
            if df_frame is None:
                return None
            if target_col in df_frame.columns:
                eprint("RUN_INFO: target_present_in", name, target_col, "rows=", len(df_frame))
                return df_frame
            if "Close" in df_frame.columns:
                try:
                    df_copy = df_frame.copy()
                    df_copy[target_col] = df_copy["Close"].shift(-1)
                    before = len(df_copy)
                    df_copy = df_copy.dropna(subset=[target_col]).reset_index(drop=True)
                    after = len(df_copy)
                    eprint("RUN_INFO: created_target_in", name, target_col, f"rows_before={before}", f"rows_after={after}")
                    return df_copy
                except Exception as e:
                    eprint("RUN_WARN: target_creation_failed_in", name, str(e))
                    return df_frame
            eprint("RUN_WARN: cannot_create_target_no_Close_in", name)
            return df_frame

        if args.train_csv and args.test_csv:
            df_train = _propagate_target(df_train, "train")
            df_val = _propagate_target(df_val, "val") if df_val is not None else None
            df_test = _propagate_target(df_test, "test")

            if df_train is None or (target_col not in df_train.columns):
                eprint("RUN_ERROR: required_column_missing", target_col, "in train")
                print("RUN_COMPLETE: FAILURE")
                return 1
            if df_test is None or (target_col not in df_test.columns):
                eprint("RUN_ERROR: required_column_missing", target_col, "in test")
                print("RUN_COMPLETE: FAILURE")
                return 1

            _required_cols_train = list(features) + ([target_col] if target_col in df_train.columns else [])
            _required_cols_val = list(features) + ([target_col] if (df_val is not None and target_col in df_val.columns) else [])
            _required_cols_test = list(features) + ([target_col] if target_col in df_test.columns else [])

            try:
                df_train = df_train.dropna(axis=0, subset=_required_cols_train).reset_index(drop=True)
                if df_val is not None:
                    df_val = df_val.dropna(axis=0, subset=_required_cols_val).reset_index(drop=True)
                df_test = df_test.dropna(axis=0, subset=_required_cols_test).reset_index(drop=True)
            except Exception as e:
                eprint("RUN_ERROR: selective_dropna_failed_in_splits", str(e))
                print("RUN_COMPLETE: FAILURE")
                return 1

            df_train = safe_cast_float(df_train, [c for c in _required_cols_train if c in df_train.columns])
            if df_val is not None:
                df_val = safe_cast_float(df_val, [c for c in _required_cols_val if c in df_val.columns])
            df_test = safe_cast_float(df_test, [c for c in _required_cols_test if c in df_test.columns])

            missing_train = [c for c in features if c not in df_train.columns]
            missing_test = [c for c in features if c not in df_test.columns]
            if missing_train:
                eprint("RUN_ERROR: missing_after_preprocessing_train", missing_train)
                print("RUN_COMPLETE: FAILURE")
                return 1
            if missing_test:
                eprint("RUN_ERROR: missing_after_preprocessing_test", missing_test)
                print("RUN_COMPLETE: FAILURE")
                return 1

        else:
            if (target_col not in df.columns) and ("Close" in df.columns):
                try:
                    df[target_col] = df["Close"].shift(-1)
                    eprint("RUN_INFO: created_target_in_combined", target_col, "rows=", len(df))
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

        # Build X/y and run model
        if args.train_csv and args.test_csv:
            X_train = df_train[features].astype(float).values
            y_train = df_train[target_col].astype(float).values
            X_val = df_val[features].astype(float).values if df_val is not None else None
            y_val = df_val[target_col].astype(float).values if df_val is not None else None
            X_test = df_test[features].astype(float).values
            y_test = df_test[target_col].astype(float).values
        else:
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

        eprint("RUN_INFO: model_path", str(model_path))
        eprint("RUN_INFO: deterministic_meta_end", json.dumps(deterministic_meta))
        print("RUN_COMPLETE: SUCCESS")
        return 0

    except Exception as exc:
        eprint("RUN_ERROR: unhandled_exception", str(exc))
        print("RUN_COMPLETE: FAILURE")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
