import pickle
from sklearn.linear_model import LinearRegression
from pathlib import Path
# calibration params observed from calibrate_preds.py
coef = [0.0609201155602932]
intercept = -3.2319717
lr = LinearRegression()
# set attributes so lr.predict works: coef_ and intercept_, and fit intercept flag
import numpy as np
lr.coef_ = np.array(coef)
lr.intercept_ = float(intercept)
lr._is_fitted = True
# For sklearn compatibility: create a fitted object by setting n_features_in_
lr.n_features_in_ = 1
# save pickle
p = Path("ho_artifact_outputs") / "target_calibrator.pkl"
with open(p, "wb") as f:
    pickle.dump(lr, f)
# save sidecar for human/machine readability
side = {
    "artifact": "target_calibrator.pkl",
    "type": "LinearRegressionAffineCalibrator",
    "coef": coef,
    "intercept": intercept,
    "created_by": "calibration-session",
    "note": "Apply as y_calibrated = coef[0] * y_pred + intercept"
}
import json
with open(Path("ho_artifact_outputs") / "target_calibrator_sidecar.json", "w") as f:
    json.dump(side, f, indent=2)
print("WRITTEN", p)
