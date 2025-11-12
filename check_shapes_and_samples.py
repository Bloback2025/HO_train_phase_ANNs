# check_shapes_and_samples.py
# Prints shapes and first few aligned samples for y_test_s, y_pred, and naive arrays
import os, json, importlib.util
import numpy as np
from tensorflow import keras
from bootstrap_ho_paths_and_patch import train_path, val_path, test_path, BASE_DIR

# dynamic import of the training module (handles filenames with dots)
module_path = os.path.join(BASE_DIR, "train_phase2b2_HO_v5_heavy_v5.1.py")
spec = importlib.util.spec_from_file_location("train_v5_1_module", module_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

load_csv = train_mod.load_csv
shift_forward = train_mod.shift_forward
mu = train_mod.mu
sigma = train_mod.sigma

dates_train, X_train, y_train = load_csv(train_path)
dates_val,   X_val,   y_val   = load_csv(val_path)
dates_test,  X_test,  y_test  = load_csv(test_path)

X_train_s, y_train_s = shift_forward(X_train, y_train)
X_val_s,   y_val_s   = shift_forward(X_val,   y_val)
X_test_s,  y_test_s  = shift_forward(X_test,  y_test)

X_test_n = (X_test_s - mu) / sigma

# load latest model
# BROKEN: def safe_load_latest_model(base_dir):
    files = [f for f in os.listdir(base_dir) if f.startswith("2bANN2_HO_model_v5.1_heavy_") and f.endswith(".keras")]
    files.sort(reverse=True)
# BROKEN:     if not files:
        return None
# BROKEN:     try:
        return keras.models.load_model(os.path.join(base_dir, files[0]))
# BROKEN:     except Exception:
        return None

model = safe_load_latest_model(BASE_DIR)
# BROKEN: if model is None:
    print(json.dumps({"warning": "No model found or failed to load. Run training first."}, indent=2))
# BROKEN: else:
    y_pred = model.predict(X_test_n, verbose=0).reshape(-1)
    out = {
        "y_test_s_shape": list(y_test_s.shape),
        "y_pred_shape": list(y_pred.shape),
        "first_5_y_test_s": [float(x) for x in y_test_s[:5].tolist()],
        "first_5_y_pred": [float(x) for x in y_pred[:5].tolist()],
        "first_5_y_pred_aligned": [float(x) for x in y_pred[1:6].tolist()],
        "first_5_y_naive": [float(x) for x in y_test_s[:-1][:5].tolist()]
    }
    print(json.dumps(out, indent=2))
