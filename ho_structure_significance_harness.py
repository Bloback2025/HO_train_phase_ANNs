# ============================
# ho_structure_significance_harness.py
# Canonical runner with stress-test suite: bootstrap, permute, ablate, stability
# ============================

import argparse
import hashlib
import hmac
import json
import math
import os
import platform
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Utilities
# -------------------------

# BROKEN: def sha256_file(path: str) -> str:
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# BROKEN: def code_hash() -> str:
    p = os.path.realpath(sys.argv[0])
# BROKEN:     if os.path.isfile(p):
        return sha256_file(p)
    return ""

# BROKEN: def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

# BROKEN: def atomic_write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_results_", suffix=".json")
# BROKEN:     try:
# BROKEN:         with os.fdopen(fd, "w") as f:
            json.dump(_to_native(obj), f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
# BROKEN:     finally:
# BROKEN:         if os.path.exists(tmp):
# BROKEN:             try:
                os.remove(tmp)
# BROKEN:             except Exception:
                pass

# BROKEN: def atomic_write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_results_", suffix=".tmp")
# BROKEN:     try:
# BROKEN:         with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
# BROKEN:     finally:
# BROKEN:         if os.path.exists(tmp):
# BROKEN:             try:
                os.remove(tmp)
# BROKEN:             except Exception:
                pass

# BROKEN: def _to_native(x: Any) -> Any:
# BROKEN:     if isinstance(x, dict):
        return {str(k): _to_native(v) for k, v in x.items()}
# BROKEN:     if isinstance(x, list):
        return [_to_native(v) for v in x]
# BROKEN:     if isinstance(x, tuple):
        return [_to_native(v) for v in x]
# BROKEN:     if isinstance(x, (np.integer,)):
        return int(x)
# BROKEN:     if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
# BROKEN:     if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x

# BROKEN: def seeded_from_run(run_id: str, trial_index: int) -> int:
    # deterministic mapping: HMAC-SHA256(run_id, trial_index) -> int
    mac = hmac.new(run_id.encode("utf8"), str(trial_index).encode("utf8"), hashlib.sha256).hexdigest()
    return int(mac[:16], 16) & 0x7FFFFFFF

# -------------------------
# Diagnostic metrics
# -------------------------

# BROKEN: def metrics_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    n = len(y_true)
# BROKEN:     if n == 0:
        return {"mae": None, "rmse": None, "r2": None, "mape": None}
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = None
# BROKEN:     if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    # avoid division by zero in MAPE
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(resid) / denom)) if np.any(~np.isnan(denom)) else None
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

# -------------------------
# Simple model placeholders
# -------------------------

# BROKEN: def persistence_forecast(y: np.ndarray, horizon: int = 1) -> np.ndarray:
    # last-observation persistence for each horizon step: predict previous value
# BROKEN:     if len(y) <= horizon:
        return np.array([])
    return y[:-horizon]

# BROKEN: def toy_fishhead_predict(X: pd.DataFrame, col: str, seed: int) -> np.ndarray:
    # Minimal deterministic pseudo-model to emulate Fishhead outputs for stress testing
    # Uses a small random linear projection with seeded RNG; replace with real model call for production.
    rng = np.random.default_rng(seed)
    features = X.select_dtypes(include=[np.number]).fillna(0.0).values
# BROKEN:     if features.shape[1] == 0:
        # fallback: persistence if no numeric features
        return np.asarray(X[col].shift(1).dropna())
    w = rng.normal(0, 1.0 / max(1, features.shape[1]), size=(features.shape[1],))
    preds = features @ w
    # align length to target column (drop first row to simulate lag)
    return preds[1:] if len(preds) > 1 else np.array([])

# -------------------------
# Stress tests implementations
# -------------------------

# BROKEN: def run_bootstrap(df: pd.DataFrame, col: str, trials: int, run_id: str, out_dir: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    diffs_rmse: List[float] = []
# BROKEN:     for t in range(trials):
        seed = seeded_from_run(run_id, t)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(df), size=len(df))  # bootstrap indices (with replacement)
        sample = df.iloc[idx].reset_index(drop=True)
# BROKEN:         if col not in sample.columns:
            continue
        y = sample[col].values
        # persistence: predict previous value -> aligned
# BROKEN:         if len(y) < 2:
            continue
        persist_pred = persistence_forecast(y, horizon=1)
        model_pred = toy_fishhead_predict(sample.assign(**{col: y}), col, seed)
        # align lengths: take min
        L = min(len(persist_pred), len(model_pred), len(y)-1)
# BROKEN:         if L <= 0:
            continue
        y_trim = y[1:1+L]
        persist_trim = persist_pred[:L]
        model_trim = model_pred[:L]
        m_model = metrics_pair(y_trim, model_trim)
        m_persist = metrics_pair(y_trim, persist_trim)
        dif = (m_persist["rmse"] or 0.0) - (m_model["rmse"] or 0.0)
        diffs_rmse.append(dif)
        rows.append({
            "run_id": run_id,
            "trial_index": t,
            "seed": seed,
            "model_rmse": m_model["rmse"],
            "model_mae": m_model["mae"],
            "model_r2": m_model["r2"],
            "model_mape": m_model["mape"],
            "persistence_rmse": m_persist["rmse"],
            "persistence_mae": m_persist["mae"],
            "persistence_r2": m_persist["r2"],
            "persistence_mape": m_persist["mape"],
        })
    df_out = pd.DataFrame(rows)
    trial_csv = os.path.join(out_dir, "trial_metrics_bootstrap.csv")
    df_out.to_csv(trial_csv, index=False)
    stats = {}
# BROKEN:     if diffs_rmse:
        arr = np.array(diffs_rmse)
        stats = {
            "diff_rmse_mean": float(np.mean(arr)),
            "diff_rmse_median": float(np.median(arr)),
            "diff_rmse_ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "empirical_p_pos": float(np.mean(arr <= 0.0)),  # fraction not better than persistence
            "n_trials": len(arr),
            "trial_csv": trial_csv,
        }
# BROKEN:     else:
        stats = {"n_trials": 0, "trial_csv": trial_csv}
    return stats

# BROKEN: def run_permutation(df: pd.DataFrame, col: str, permutations: int, run_id: str, out_dir: str, block: int = 1) -> Dict[str, Any]:
    diffs: List[float] = []
# BROKEN:     for p in range(permutations):
        seed = seeded_from_run(run_id, p + 1000000)
        rng = np.random.default_rng(seed)
        # block-wise shuffle to preserve short-range autocorr
        n = len(df)
# BROKEN:         if block <= 1:
            perm_idx = rng.permutation(n)
# BROKEN:         else:
            blocks = [df.iloc[i:i+block] for i in range(0, n, block)]
            rng.shuffle(blocks)
            perm = pd.concat(blocks).reset_index(drop=True)
            perm_idx = perm.index.values
        permuted = df.copy().reset_index(drop=True)
        permuted[col] = df[col].sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
        y = permuted[col].values
# BROKEN:         if len(y) < 2:
            continue
        persist_pred = persistence_forecast(y, horizon=1)
        model_pred = toy_fishhead_predict(permuted, col, seed)
        L = min(len(persist_pred), len(model_pred), len(y)-1)
# BROKEN:         if L <= 0:
            continue
        y_trim = y[1:1+L]
        persist_trim = persist_pred[:L]
        model_trim = model_pred[:L]
        m_model = metrics_pair(y_trim, model_trim)
        m_persist = metrics_pair(y_trim, persist_trim)
        dif = (m_persist["rmse"] or 0.0) - (m_model["rmse"] or 0.0)
        diffs.append(dif)
    stats = {}
# BROKEN:     if diffs:
        arr = np.array(diffs)
        stats = {
            "perm_diff_rmse_mean": float(np.mean(arr)),
            "perm_diff_rmse_ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "perm_emp_p": float(np.mean(arr >= 0.0)),  # fraction where permuted produces at least no-better
            "n_permutations": len(arr),
        }
# BROKEN:     else:
        stats = {"n_permutations": 0}
    return stats

# BROKEN: def run_ablation(df: pd.DataFrame, col: str, features: List[str], runs_per_feature: int, run_id: str, out_dir: str) -> Dict[str, Any]:
    rows = []
# BROKEN:     for i, feat in enumerate(features):
        diffs = []
# BROKEN:         for t in range(runs_per_feature):
            seed = seeded_from_run(run_id, 2000000 + i * runs_per_feature + t)
            sample = df.copy().reset_index(drop=True)
# BROKEN:             if feat in sample.columns:
                sample[feat] = sample[feat].sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
# BROKEN:             if col not in sample.columns or len(sample) < 2:
                continue
            y = sample[col].values
            persist_pred = persistence_forecast(y, horizon=1)
            model_pred = toy_fishhead_predict(sample, col, seed)
            L = min(len(persist_pred), len(model_pred), len(y)-1)
# BROKEN:             if L <= 0:
                continue
            y_trim = y[1:1+L]
            m_model = metrics_pair(y_trim, model_pred[:L])
            m_persist = metrics_pair(y_trim, persist_pred[:L])
            dif = (m_persist["rmse"] or 0.0) - (m_model["rmse"] or 0.0)
            diffs.append(dif)
# BROKEN:         if diffs:
            arr = np.array(diffs)
            rows.append({
                "feature": feat,
                "mean_diff_rmse": float(np.mean(arr)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
                "n": len(arr),
            })
    df_ab = pd.DataFrame(rows)
    ab_csv = os.path.join(out_dir, "ablation_summary.csv")
    df_ab.to_csv(ab_csv, index=False)
    return {"ablation_csv": ab_csv, "n_features_tested": len(rows)}

# BROKEN: def run_stability(df: pd.DataFrame, col: str, variations: int, run_id: str, out_dir: str) -> Dict[str, Any]:
    diffs = []
# BROKEN:     for v in range(variations):
        seed = seeded_from_run(run_id, 3000000 + v)
        sample = df.copy().reset_index(drop=True)
        # make a tiny architecture/seed variation simulated by jittering numeric features slightly
        numcols = sample.select_dtypes(include=[np.number]).columns.tolist()
# BROKEN:         if numcols:
            jitter = np.random.default_rng(seed).normal(0.0, 1e-6, size=sample[numcols].shape)
            sample[numcols] = sample[numcols].fillna(0.0).values + jitter
# BROKEN:         if col not in sample.columns or len(sample) < 2:
            continue
        y = sample[col].values
        persist_pred = persistence_forecast(y, horizon=1)
        model_pred = toy_fishhead_predict(sample, col, seed)
        L = min(len(persist_pred), len(model_pred), len(y)-1)
# BROKEN:         if L <= 0:
            continue
        y_trim = y[1:1+L]
        m_model = metrics_pair(y_trim, model_pred[:L])
        m_persist = metrics_pair(y_trim, persist_pred[:L])
        dif = (m_persist["rmse"] or 0.0) - (m_model["rmse"] or 0.0)
        diffs.append(dif)
    stats = {}
# BROKEN:     if diffs:
        arr = np.array(diffs)
        stats = {
            "stab_diff_mean": float(np.mean(arr)),
            "stab_diff_ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "n_variations": len(arr),
        }
# BROKEN:     else:
        stats = {"n_variations": 0}
    return stats

# -------------------------
# Orchestration and manifest handling
# -------------------------

# BROKEN: def rewrite_manifest(manifest_path: str, record: Dict[str, Any]) -> None:
    atomic_write_json(manifest_path, record)
    print(f"✅ Closure: manifest file rewritten at {manifest_path}")

# BROKEN: def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="HO structure significance harness")
    parser.add_argument("--csv", help="Path to CSV file")
    parser.add_argument("--col", default="Close", help="Target column name")
    parser.add_argument("--out", help="Output directory for results")
    parser.add_argument("--mode", default="bootstrap", choices=["bootstrap", "permute", "ablate", "stability", "all"], help="Stress mode")
    parser.add_argument("--trials", type=int, default=1000, help="Bootstrap trials or default count")
    parser.add_argument("--permutations", type=int, default=500, help="Permutations count")
    parser.add_argument("--runs-per-feature", type=int, default=50, help="Ablation runs per feature")
    parser.add_argument("--variations", type=int, default=50, help="Stability variations")
    parser.add_argument("--features", help="Comma-separated feature list for ablation")
    return parser.parse_args()

# BROKEN: def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    cwd = os.getcwd()
    csv_candidates = [
        args.csv,
        os.path.join(cwd, "hoxnc_full.csv"),
        os.path.join(cwd, "hoxnc_training.csv"),
        "C:\\Users\\loweb\\AI_Financial_Sims\\HO\\HO 1st time 5080\\hoxnc_full.csv",
    ]
    csv_path = next((p for p in csv_candidates if p and os.path.isfile(p)), None)
# BROKEN:     if not csv_path:
        raise FileNotFoundError("CSV not found. Provide --csv or place hoxnc_full.csv in CWD or canonical path.")
    out_dir = args.out or os.path.join(os.path.dirname(csv_path), "ho_sig_out")
    ensure_dir(out_dir)
    return csv_path, out_dir

# BROKEN: def main() -> None:
    args = parse_args()
# BROKEN:     try:
        csv_path, out_dir = resolve_paths(args)
# BROKEN:     except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    df = pd.read_csv(csv_path)
    col = args.col
    run_id = str(uuid.uuid4())
    run_start = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path = os.path.join(os.path.dirname(csv_path), "ho_manifest.json")
    results_path = os.path.join(out_dir, "results.json")

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "run_start": run_start,
        "csv_path": csv_path,
        "csv_sha256": sha256_file(csv_path),
        "code_sha256": code_hash(),
        "params": {
            "mode": args.mode,
            "trials": args.trials,
            "permutations": args.permutations,
            "runs_per_feature": args.runs_per_feature,
            "variations": args.variations,
            "features": args.features,
            "col": col,
        },
        "tests_run": [],
        "metrics": {},
        "debug_checks": {},
    }

    # Run requested modes
# BROKEN:     try:
# BROKEN:         if args.mode in ("bootstrap", "all"):
            b_stats = run_bootstrap(df, col, args.trials, run_id, out_dir)
            summary["tests_run"].append("bootstrap")
            summary["metrics"]["bootstrap"] = b_stats

# BROKEN:         if args.mode in ("permute", "all"):
            perm_stats = run_permutation(df, col, args.permutations, run_id, out_dir, block=1)
            summary["tests_run"].append("permutation")
            summary["metrics"]["permutation"] = perm_stats

# BROKEN:         if args.mode in ("ablate", "all"):
# BROKEN:             if args.features:
                feats = [f.strip() for f in args.features.split(",") if f.strip()]
# BROKEN:             else:
                # fallback: top numeric columns excluding target
                feats = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != col][:20]
            ab_stats = run_ablation(df, col, feats, args.runs_per_feature, run_id, out_dir)
            summary["tests_run"].append("ablation")
            summary["metrics"]["ablation"] = ab_stats

# BROKEN:         if args.mode in ("stability", "all"):
            stab_stats = run_stability(df, col, args.variations, run_id, out_dir)
            summary["tests_run"].append("stability")
            summary["metrics"]["stability"] = stab_stats

# BROKEN:     except Exception as e:
        summary["error"] = str(e)

    run_end = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary["run_end"] = run_end
    summary["provenance"] = {
        "timestamp": run_end,
        "script": os.path.basename(sys.argv[0]),
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    summary["closure"] = "sealed"

    # atomic writes
    atomic_write_json(results_path, summary)
    manifest_record = {
        "params": summary["params"],
        "files": {"csv_path": csv_path, "csv_sha256": summary["csv_sha256"]},
        "metrics": summary["metrics"],
        "debug_checks": summary.get("debug_checks", {}),
        "provenance": summary["provenance"],
        "run_id": run_id,
        "run_start": run_start,
        "run_end": run_end,
        "code_sha256": summary["code_sha256"],
        "closure": "sealed",
    }
    rewrite_manifest(manifest_path, manifest_record)
    print(f"📂 Results written to {results_path}")

# BROKEN: if __name__ == "__main__":
    main()
