import os
import datetime
import hashlib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- Bootstrap paths ---
from bootstrap_ho_paths_and_patch import BASE_DIR, train_path, val_path, test_path

# --- Utility: hash a file ---
# BROKEN: def hash_file(path):
    h = hashlib.sha256()
# BROKEN:     with open(path, "rb") as f:
# BROKEN:         for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# --- Load dataset ---
# BROKEN: def load_dataset(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.capitalize()
    df = df.sort_values("Date")
    dates = df["Date"].values
    X = df[["Open", "High", "Low"]].values.astype(float)
    y = df["Close"].values.astype(float)
    return dates, X, y

# --- Load data ---
dates_train, X_train, y_train = load_dataset(train_path)
dates_val,   X_val,   y_val   = load_dataset(val_path)
dates_test,  X_test,  y_test  = load_dataset(test_path)

# --- Build model ---
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(3,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=2
)

# --- Evaluate ANN ---
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test).flatten()
r2_ann = r2_score(y_test, y_pred)

print(f"[RESULT] Test Loss={test_loss:.6f}, Test MAE={test_mae:.6f}, R²={r2_ann:.6f}")

# --- Naïve baseline ---
y_naive = y_test[:-1]
y_true  = y_test[1:]
mae_naive = mean_absolute_error(y_true, y_naive)
r2_naive  = r2_score(y_true, y_naive)
print(f"[BASELINE] Naïve MAE={mae_naive:.6f}, R²={r2_naive:.6f}")

# --- Save model ---
model_file = os.path.join(BASE_DIR, "2bANN2_HO_model.keras")
model.save(model_file)
print(f"[SAVED] {model_file}")

# --- Log run ---
meta = {
    "script": "train_phase2b2_HO_v2.py",
    "train_file": train_path,
    "val_file": val_path,
    "test_file": test_path,
    "train_hash": hash_file(train_path),
    "val_hash": hash_file(val_path),
    "test_hash": hash_file(test_path),
    "model_file": model_file,
    "model_hash": hash_file(model_file),
    "test_loss": float(test_loss),
    "test_mae": float(test_mae),
    "r2_ann": float(r2_ann),
    "mae_naive": float(mae_naive),
    "r2_naive": float(r2_naive),
    "timestamp": datetime.datetime.now().isoformat()
}

log_file = os.path.join(BASE_DIR, f"RUNLOG_2bANN2_HO_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
import json
# BROKEN: with open(log_file, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[LOGGED] {log_file}")

# --- Optional: plot training history ---
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()
