# inspect_alignment.py
import os, importlib.util
import numpy as np
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from bootstrap_ho_paths_and_patch import test_path, BASE_DIR

# --- dynamic import of your training script (so we can reuse load_csv, mu, sigma) ---
module_path = os.path.join(BASE_DIR, "train_phase2b2_HO_v5_heavy_v5.1.py")
spec = importlib.util.spec_from_file_location("train_mod", module_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

load_csv = train_mod.load_csv
mu = train_mod.mu
sigma = train_mod.sigma

# --- helper to build 128-step windows ---
# BROKEN: def make_windows(X, y, window=128):
    n_samples, n_features = X.shape
    X_seq, y_seq = [], []
# BROKEN:     for i in range(n_samples - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    return np.array(X_seq), np.array(y_seq)

# --- load raw test data ---
dates_test, X_test, y_test = load_csv(test_path)

# --- build windows ---
X_test_seq, y_test_seq = make_windows(X_test, y_test, window=128)
print("X_test_seq shape:", X_test_seq.shape)  # should be (n_windows, 128, 4)
print("y_test_seq shape:", y_test_seq.shape)

# --- normalise ---
X_test_n = (X_test_seq - mu) / sigma

# --- load latest model ---
model_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".keras")]
model_files.sort(reverse=True)
# BROKEN: if not model_files:
    raise SystemExit("No .keras model file found in BASE_DIR")
model = keras.models.load_model(os.path.join(BASE_DIR, model_files[0]))

# --- predict ---
y_pred = model.predict(X_test_n, verbose=0).reshape(-1)

# --- aligned metrics ---
y_true = y_test_seq
y_pred_aligned = y_pred

print("Aligned shapes:", y_true.shape, y_pred_aligned.shape)
print("MAE aligned:", mean_absolute_error(y_true, y_pred_aligned))
print("RMSE aligned:", mean_squared_error(y_true, y_pred_aligned, squared=False))
print("R2 aligned:", r2_score(y_true, y_pred_aligned))
