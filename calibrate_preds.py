import os, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# load preds (prefer scaled preds saved earlier), fallback to model predict run
if os.path.exists("nulltest_preds.npy"):
    preds = np.load("nulltest_preds.npy")
elif os.path.exists("nulltest_preds_scaled.npy"):
    preds = np.load("nulltest_preds_scaled.npy")
else:
    # produce preds now
    from tensorflow.keras.models import load_model
    m = load_model(r"ho_artifact_outputs\2bANN2_HO_model.keras")
    X = np.load("X_eval_scaled.npy")
    preds = m.predict(X).reshape(-1)
    np.save("nulltest_preds_scaled.npy", preds)

y = np.load("y_eval.npy")
n = len(y)
k = int(n * 0.8)
Xfit = preds[:k].reshape(-1,1)
yfit = y[:k]
Xval = preds[k:].reshape(-1,1)
yval = y[k:]

lr = LinearRegression().fit(Xfit,yfit)
ycal_val = lr.predict(Xval)
print("CALIBRATION coef,intercept:", lr.coef_.tolist(), lr.intercept_)
print("VAL R2:", r2_score(yval,ycal_val))
print("VAL MSE:", mean_squared_error(yval,ycal_val))
print("VAL MAE:", mean_absolute_error(yval,ycal_val))

# full calibrated preds saved
ycal_full = lr.predict(preds.reshape(-1,1))
np.save("nulltest_preds_calibrated.npy", ycal_full)
print("SAVED nulltest_preds_calibrated.npy")
