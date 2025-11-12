# audit_fishhead_all.py
import os, importlib.util
import numpy as np
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bootstrap_ho_paths_and_patch import test_path, BASE_DIR

# --- import training module to reuse load_csv, mu, sigma ---
module_path = os.path.join(BASE_DIR, "train_phase2b2_HO_v5_heavy_v5.1.py")
spec = importlib.util.spec_from_file_location("train_mod", module_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

load_csv = train_mod.load_csv
mu = train_mod.mu
sigma = train_mod.sigma

# --- helper: build 128-step windows ---
# BROKEN: def make_windows(X, y, window=128):
    n_samples, n_features = X.shape
    X_seq, y_seq = [], []
# BROKEN:     for i in range(n_samples - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

# --- load test data ---
_, X_test, y_test = load_csv(test_path)
X_test_seq, y_test_seq = make_windows(X_test, y_test, window=128)
X_test_n = (X_test_seq - mu) / sigma

print("X_test_seq shape:", X_test_seq.shape)
print("y_test_seq shape:", y_test_seq.shape)

# --- find all Fishhead models ---
model_files = [f for f in os.listdir(BASE_DIR) if f.startswith("Fishhead") and f.endswith(".keras")]
model_files.sort()  # chronological order

# BROKEN: if not model_files:
    raise SystemExit("No Fishhead .keras models found in BASE_DIR")

# --- audit each model ---
# BROKEN: for fname in model_files:
    model_path = os.path.join(BASE_DIR, fname)
    print("\n=== Auditing model:", fname, "===")
    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test_n, verbose=0).reshape(-1)

    mae = mean_absolute_error(y_test_seq, y_pred)
    rmse = mean_squared_error(y_test_seq, y_pred, squared=False)
    r2 = r2_score(y_test_seq, y_pred)

    print("First 3 y_true:", y_test_seq[:3])
    print("First 3 y_pred:", y_pred[:3])
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
