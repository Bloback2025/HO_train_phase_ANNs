import os, sys, pickle, numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

ART = Path("ho_artifact_outputs")
MODEL = ART / "2bANN2_HO_model.keras"
CAL = ART / "target_calibrator.pkl"
Xf = Path("X_eval_scaled.npy")
yf = Path("y_eval.npy")

def fail(msg):
    print("DIAG_FAIL:", msg)
    sys.exit(2)

if not MODEL.exists():
    fail(f"model missing: {MODEL}")
if not Xf.exists():
    fail(f"X_eval_scaled.npy missing")
if not yf.exists():
    fail(f"y_eval.npy missing")

m = load_model(str(MODEL))
X = np.load(str(Xf))
y = np.load(str(yf)).reshape(-1)

pred = m.predict(X).reshape(-1)
print("SHAPES: X", X.shape, "pred", pred.shape, "y", y.shape)
print("UNCALIBRATED STATS: pred mean/std", float(pred.mean()), float(pred.std()), "y mean/std", float(y.mean()), float(y.std()))
print("UNCALIBRATED R2:", r2_score(y, pred), "MSE:", mean_squared_error(y,pred), "MAE:", mean_absolute_error(y,pred))

if CAL.exists():
    print("FOUND calibrator:", CAL)
    cal = pickle.load(open(str(CAL),"rb"))
    try:
        pred_cal = cal.predict(pred.reshape(-1,1)).reshape(-1)
        print("APPLIED calibrator via predict()")
    except Exception as e:
        try:
            pred_cal = cal.predict(pred)
            print("APPLIED calibrator via direct predict()")
        except Exception as e2:
            print("CALIBRATOR APPLY FAILED:", e, e2)
            pred_cal = None
    if pred_cal is not None:
        print("CALIBRATED STATS: pred_cal mean/std", float(pred_cal.mean()), float(pred_cal.std()))
        print("CALIBRATED R2:", r2_score(y, pred_cal), "MSE:", mean_squared_error(y,pred_cal), "MAE:", mean_absolute_error(y,pred_cal))
        # show small sample heads for manual inspection
        n = min(10, len(y))
        print("SAMPLE (y, pred_uncal, pred_cal) first", n)
        for i in range(n):
            print(i, float(y[i]), float(pred[i]), float(pred_cal[i]))
        np.save("diag_pred_uncal.npy", pred)
        np.save("diag_pred_cal.npy", pred_cal)
        print("SAVED diag_pred_uncal.npy, diag_pred_cal.npy")
else:
    print("NO calibrator found at", CAL)
