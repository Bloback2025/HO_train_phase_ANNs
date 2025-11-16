import numpy as np
import json, os, sys
from pathlib import Path
try:
    # locate artifact folder and sidecars
    art = Path(r"ho_artifact_outputs")
    sidecar_j = art / "scaler_sidecar.json"
    pkl = art / "scaler.pkl"
    target_scaler = None
    input_scaler = None
    if sidecar_j.exists():
        sc_meta = json.load(open(sidecar_j, "r"))
        # sidecar keys: means, scales, feature_names, scaler_class
        # prefer pickle if present
    if pkl.exists():
        import pickle
        input_scaler = pickle.load(open(pkl,"rb"))
    # load X/y if not already loaded by the script caller
    if "X_eval" not in globals():
        if Path("X_eval.npy").exists():
            X_eval = np.load("X_eval.npy")
        elif Path("X_eval_scaled.npy").exists():
            X_eval = np.load("X_eval_scaled.npy")
        else:
            raise FileNotFoundError("X_eval.npy or X_eval_scaled.npy not found")
    if "y_eval" not in globals():
        if Path("y_eval.npy").exists():
            y_eval = np.load("y_eval.npy")
        else:
            raise FileNotFoundError("y_eval.npy not found")
    # apply input scaler if script called with raw X and scaler exists
    if input_scaler is not None and ("X_eval_scaled" not in globals()):
        try:
            X_eval_scaled = input_scaler.transform(X_eval)
            X_used = X_eval_scaled
        except Exception:
            X_used = X_eval
    else:
        X_used = X_eval
except Exception as e:
    # fail fast and keep original traceback visible
    print("MANIFEST/PREPROCESS GUARD WARNING:", e)
    X_used = globals().get("X_eval", None)
    y_eval = globals().get("y_eval", None)
    input_scaler = None
    target_scaler = None
from tensorflow.keras.models import load_model
X_eval = np.load("X_eval.npy")
y_eval = np.load("y_eval.npy")
model = load_model(r"ho_artifact_outputs\2bANN2_HO_model.keras")
X_eval = np.load("X_eval.npy")
y_eval = np.load("y_eval.npy")
from sklearn.metrics import r2_score
# replace these with your actual arrays loaded from your harness
# X_eval, y_eval, model are expected to be available when you run this script
# Example placeholders (remove if you load real data):
# X_eval = np.load("X_eval.npy")
# y_eval = np.load("y_eval.npy")
# import tensorflow as tf; model = tf.keras.models.load_model("2bANN2_HO_model.keras")

# NULL-INPUT
X_rand = np.random.normal(size=X_eval.shape)
y_pred = model.predict(X_rand).reshape(-1)
print("null-input R2:", r2_score(y_eval, y_pred))

# NULL-TARGET (shuffle)
y_shuf = np.random.permutation(y_eval)
y_pred = model.predict(X_eval).reshape(-1)
print("null-target (shuffled) R2:", r2_score(y_shuf, y_pred))

# FULLY-RANDOM
X_rand = np.random.normal(size=X_eval.shape)
y_rand = np.random.normal(size=y_eval.shape)
y_pred = model.predict(X_rand).reshape(-1)
print("fully-random R2:", r2_score(y_rand, y_pred))
