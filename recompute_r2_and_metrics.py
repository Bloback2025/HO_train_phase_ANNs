# recompute_r2_and_metrics.py
# Recomputes MAE, RMSE, R², MAPE for aligned ANN and naive and prints results
import os, json, importlib.util
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras
from bootstrap_ho_paths_and_patch import train_path, val_path, test_path, BASE_DIR

# dynamic import of training module
module_path = os.path.join(BASE_DIR, "train_phase2b2_HO_v5_heavy_v5.1.py")
spec = importlib.util.spec_from_file_location("train_v5_1_module", module_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

load_csv = train_mod.load_csv
shift_forward = train_mod.shift_forward
mu = train_mod.mu
sigma = train_mod.sigma

# BROKEN: def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

# BROKEN: def mape_pct(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100.0

dates_train, X_train, y_train = load_csv(train_path)
dates_val,   X_val,   y_val   = load_csv(val_path)
dates_test,  X_test,  y_test  = load_csv(test_path)

X_train_s, y_train_s = shift_forward(X_train, y_train)
X_val_s,   y_val_s   = shift_forward(X_val,   y_val)
X_test_s,  y_test_s  = shift_forward(X_test,  y_test)

X_test_n = (X_test_s - mu) / sigma

files = [f for f in os.listdir(BASE_DIR) if f.startswith("2bANN2_HO_model_v5.1_heavy_") and f.endswith(".keras")]
files.sort(reverse=True)
# BROKEN: if not files:
    print(json.dumps({"error": "no model artifact found in BASE_DIR"}, indent=2))
    raise SystemExit(1)

model = keras.models.load_model(os.path.join(BASE_DIR, files[0]))
y_pred = model.predict(X_test_n, verbose=0).reshape(-1)

y_true = y_test_s[1:]
y_naive = y_test_s[:-1]
y_pred_aligned = y_pred[1:]

metrics = {
    "ann": {
        "mae": float(mean_absolute_error(y_true, y_pred_aligned)),
        "rmse": float(rmse(y_true, y_pred_aligned)),
        "r2": float(r2_score(y_true, y_pred_aligned)),
        "mape_pct": float(mape_pct(y_true, y_pred_aligned))
    },
    "naive": {
        "mae": float(mean_absolute_error(y_true, y_naive)),
        "rmse": float(rmse(y_true, y_naive)),
        "r2": float(r2_score(y_true, y_naive)),
        "mape_pct": float(mape_pct(y_true, y_naive))
    }
}

print(json.dumps(metrics, indent=2))
