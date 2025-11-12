#!/usr/bin/env python3
"""
run_ann_stresstest.py
Atomic, self-contained stress test harness comparing canonical vs reconstructed ANN.

Outputs (OUTDIR):
 - reconstructed_ann.keras
 - stresstest_results_<ts>.json
 - stresstest_manifest_<ts>.json
 - artifact_hashes_summary.json

Run:
  python run_ann_stresstest.py
"""
import os, sys, json, time, hashlib, traceback, random
from datetime import datetime
import numpy as np

# Dependency check
try:
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception as e:
    print("DEPENDENCY_ERROR", str(e))
    sys.exit(2)

# ---------- CONFIG ----------
PROJECT = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080"
CANONICAL_MODEL = os.path.join(PROJECT, "2bANN2_HO_model.keras")
OUTDIR = os.path.join(PROJECT, "ann_stresstest_outputs")
SEED = 20251109
TRIALS_PER_TASK = 5      # change this value if desired
BATCH = 256
MAX_SAMPLES = 5000
# ---------- END CONFIG ----------

os.makedirs(OUTDIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # compatibility-safe RMSE
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

# ---------- Load canonical model ----------
if not os.path.exists(CANONICAL_MODEL):
    print("LOAD_FAILED", CANONICAL_MODEL)
    sys.exit(3)

try:
    base_model = tf.keras.models.load_model(CANONICAL_MODEL)
except Exception as e:
    print("LOAD_FAILED", str(e))
    traceback.print_exc()
    sys.exit(4)

# Determine input shape and n_features robustly
try:
    input_shape = base_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    n_features = input_shape[-1] if input_shape is not None else 10
    if n_features is None:
        n_features = 10
except Exception:
    n_features = 10

n_samples = min(MAX_SAMPLES, 5000)

# ---------- Reconstruct a new model with similar topology ----------
def reconstruct_model_from(base_model, n_features):
    try:
        inp_shape = (n_features,)
        inp = tf.keras.Input(shape=inp_shape, name="stresstest_input")
        x = inp
        for layer in base_model.layers:
            cls = layer.__class__.__name__
            try:
                cfg = layer.get_config()
            except Exception:
                cfg = {}
            if cls == "InputLayer":
                continue
            if cls == "Dense":
                units = cfg.get("units", None)
                activation = cfg.get("activation", None)
                if units is None:
                    try:
                        units = int(layer.output_shape[-1])
                    except Exception:
                        units = 32
                x = tf.keras.layers.Dense(units, activation=activation)(x)
            elif cls == "Dropout":
                rate = cfg.get("rate", 0.0)
                x = tf.keras.layers.Dropout(rate)(x)
            elif cls in ("BatchNormalization", "BatchNorm"):
                x = tf.keras.layers.BatchNormalization()(x)
            elif cls == "Activation":
                act = cfg.get("activation", None)
                if act:
                    x = tf.keras.layers.Activation(act)(x)
            else:
                L = getattr(tf.keras.layers, cls, None)
                if L is not None:
                    try:
                        params = {}
                        for k,v in cfg.items():
                            if isinstance(v, (str,int,float,bool)):
                                params[k] = v
                        x = L(**params)(x)
                    except Exception:
                        pass
                else:
                    pass
        model = tf.keras.Model(inputs=inp, outputs=x, name="reconstructed_ann")
        model.compile(optimizer="adam", loss="mse")
        return model
    except Exception as e:
        print("RECONSTRUCT_FAILED", str(e))
        traceback.print_exc()
        return None

new_model = reconstruct_model_from(base_model, n_features)
if new_model is None:
    print("RECONSTRUCT_FAILED")
    sys.exit(5)

# Save reconstructed model (untrained)
recon_path = os.path.join(OUTDIR, "reconstructed_ann.keras")
try:
    new_model.save(recon_path, include_optimizer=False)
except Exception:
    try:
        new_model.save(recon_path + ".h5", include_optimizer=False)
        recon_path = recon_path + ".h5"
    except Exception as e:
        print("SAVE_RECON_FAILED", str(e))
        traceback.print_exc()
        sys.exit(6)

# ---------- Synthetic data generators (robust) ----------
def base_feature_generator(n, m):
    t = np.arange(n)
    X = np.zeros((n, m), dtype=float)
    for i in range(m):
        freq = 0.005 * (1 + (i % 5))
        phase = (i * 0.31) % (2 * np.pi)
        trend = 0.0003 * (1 + (i % 3)) * t
        signal = np.sin(2*np.pi*freq*t + phase) + trend
        noise = np.random.normal(scale=0.04*(1+0.05*i), size=n)
        X[:, i] = signal + noise
    return X

def base_target_generator(X):
    m = X.shape[1]
    coeffs = np.zeros(m)
    pattern = [0,1,2,3,4,5,6,7,8,9]
    weights = [0.5, -0.3, 0.2, 0.1, 0.05, 0.04, 0.02, 0.01, 0.005, 0.15]
    for idx, w in zip(pattern, weights):
        if idx < m:
            coeffs[idx] = w
    for i in range(m):
        if coeffs[i] == 0:
            coeffs[i] = 0.01 / (1 + i)
    nonlin = np.tanh(X[:, 0] * 0.28 + (X[:, 2] * 0.12 if m > 2 else 0.0))
    y = X.dot(coeffs) + nonlin + np.random.normal(scale=0.01, size=X.shape[0])
    return y

# ---------- Stress task transforms ----------
def task_nominal(X, y): return X, y
def task_additive_noise(X, y, sigma=0.5): return X + np.random.normal(scale=sigma, size=X.shape), y
def task_missing(X, y, frac=0.2):
    Xc = X.copy()
    ncols = Xc.shape[1]
    k = max(1, int(ncols * frac))
    cols = np.random.choice(ncols, k, replace=False)
    Xc[:, cols] = 0.0
    return Xc, y
def task_scale(X, y, scale=10.0): return X * scale, y
def task_drift(X, y, strength=0.5):
    y2 = y.copy()
    cut = len(y2)//2
    y2[cut:] = y2[cut:] + strength * np.sin(np.linspace(0, 3, len(y2)-cut))
    return X, y2
def task_adv(X, y, eps=0.05):
    sign = np.sign(np.mean(X, axis=0))
    return X + eps * sign, y
def task_reorder(X, y):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    return X[idx], y[idx]
def task_shock(X, y, prob=0.01, scale=5.0):
    Xs = X.copy()
    n, m = Xs.shape
    shocks = np.random.rand(n, m) < prob
    Xs[shocks] += np.random.normal(scale=scale, size=np.count_nonzero(shocks))
    return Xs, y

TASKS = [
    ("nominal", task_nominal),
    ("add_noise_small", lambda X,y: task_additive_noise(X,y, sigma=0.1)),
    ("add_noise_large", lambda X,y: task_additive_noise(X,y, sigma=1.0)),
    ("missing_features", lambda X,y: task_missing(X,y, frac=0.3)),
    ("scale_mismatch", lambda X,y: task_scale(X,y, scale=8.0)),
    ("concept_drift", lambda X,y: task_drift(X,y, strength=0.8)),
    ("adversarial", lambda X,y: task_adv(X,y, eps=0.1)),
    ("temporal_reorder", task_reorder),
    ("heavy_tail_shock", lambda X,y: task_shock(X,y, prob=0.02, scale=6.0)),
]

# ---------- Attempt to load canonical scaler if present ----------
scaler = None
scaler_path = os.path.join(PROJECT, "scaler.pkl")
if os.path.exists(scaler_path):
    try:
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None

# ---------- Stress testing loop ----------
results = {}
print("STRESSTEST_START", datetime.utcnow().isoformat() + "Z")
for task_name, transform in TASKS:
    results[task_name] = []
    for trial in range(TRIALS_PER_TASK):
        seed = SEED + abs(hash((task_name, trial))) % 1000000
        np.random.seed(seed); random.seed(seed); tf.random.set_seed(seed)
        X = base_feature_generator(n_samples, n_features)
        y = base_target_generator(X)
        X_t, y_t = transform(X, y)
        X_scaled = X_t
        if scaler is not None:
            try:
                X_scaled = scaler.transform(X_t)
            except Exception:
                X_scaled = X_t
        try:
            y_pred_base = base_model.predict(X_scaled, batch_size=BATCH, verbose=0).reshape(-1)
        except Exception as e:
            print("PREDICT_BASE_FAILED", task_name, trial, str(e))
            y_pred_base = np.zeros_like(y_t)
        try:
            y_pred_recon = new_model.predict(X_scaled, batch_size=BATCH, verbose=0).reshape(-1)
        except Exception as e:
            print("PREDICT_RECON_FAILED", task_name, trial, str(e))
            y_pred_recon = np.zeros_like(y_t)
        m_base = metrics(y_t, y_pred_base)
        m_recon = metrics(y_t, y_pred_recon)
        results[task_name].append({
            "trial": int(trial),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_samples": int(n_samples),
            "metrics_canonical": m_base,
            "metrics_reconstructed": m_recon
        })
        print(f"TASK {task_name} TRIAL {trial} BASE_MAE {m_base['MAE']:.6f} RECON_MAE {m_recon['MAE']:.6f}")

# ---------- Save artifacts, manifest and hashes ----------
ts = int(time.time())
results_path = os.path.join(OUTDIR, f"stresstest_results_{ts}.json")
manifest_path = os.path.join(OUTDIR, f"stresstest_manifest_{ts}.json")
save_json_atomic(results_path, results)
manifest = {
    "run_started": datetime.utcnow().isoformat() + "Z",
    "seed": SEED,
    "trials_per_task": TRIALS_PER_TASK,
    "n_samples": n_samples,
    "n_features": n_features,
    "tasks": [t for t,_ in TASKS],
    "canonical_model_path": CANONICAL_MODEL,
    "reconstructed_model_path": recon_path,
    "scaler_path": scaler_path if scaler is not None else None,
    "notes": "Stress test canonical vs reconstructed (untrained) ANN on synthetic tasks"
}
save_json_atomic(manifest_path, manifest)

artifacts = {}
for p in [results_path, manifest_path, recon_path]:
    if os.path.exists(p):
        artifacts[os.path.basename(p)] = sha256_file(p)
artifact_hashes_path = os.path.join(OUTDIR, "artifact_hashes_summary.json")
save_json_atomic(artifact_hashes_path, artifacts)

print("STRESSTEST_COMPLETE")
print("RESULTS:", results_path)
print("MANIFEST:", manifest_path)
print("ARTIFACT_HASHES:", artifact_hashes_path)
for name, h in artifacts.items():
    print("HASH", name, h)
