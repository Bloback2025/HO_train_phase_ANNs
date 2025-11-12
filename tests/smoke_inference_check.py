import json, os, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
import pickle

ROOT = Path(".")
ART = ROOT / "ho_artifact_outputs"
MODEL = ART / "2bANN2_HO_model.keras"
SCALER_PKL = ART / "scaler.pkl"
SIDECAR = ART / "scaler_sidecar.json"

def fail(msg):
    print("SMOKE_FAIL:", msg)
    sys.exit(2)

if not MODEL.exists():
    fail(f"Model not found at {MODEL}")
# load sample X/y from repo canonical files
if not Path("X_eval.npy").exists() or not Path("y_eval.npy").exists():
    fail("X_eval.npy or y_eval.npy missing in repo root")
X = np.load("X_eval.npy")
y = np.load("y_eval.npy")
# apply input scaler if present
if SCALER_PKL.exists():
    sc = pickle.load(open(SCALER_PKL,"rb"))
    Xs = sc.transform(X)
else:
    Xs = X
# reduce to small sample for CI speed
N = min(128, Xs.shape[0])
Xs_short = Xs[:N]
y_short = y[:N]
m = load_model(str(MODEL))
pred = m.predict(Xs_short).reshape(-1)
# If sidecar declares target scaling (not mandatory), attempt inverse
inv_applied = False
if SIDECAR.exists():
    meta = json.load(open(SIDECAR))
    if meta.get("saved_at"):
        # if sidecar exists but no explicit target info, do not inverse by default
        pass
# compute R2 and fail if highly negative
r2 = r2_score(y_short, pred)
print("SMOKE R2:", r2)
if not np.isfinite(r2):
    fail("R2 is not finite")
if r2 < -0.5:
    fail(f"R2 below threshold: {r2}")
print("SMOKE PASS")
