import json, numpy as np, pandas as pd, tensorflow as tf
m = json.load(open(r".\manifest_2bANN.2HO.json"))
features = m["features"]; target = m["target"]
df = pd.read_csv(r".\hoxnc_testing_with_target.csv")
X = df[features].astype(float).values
y = df[target].astype(float).values
model = tf.keras.models.load_model(r".\models\2bANNa_variant0_seed20251030.keras")
pred = model.predict(X, verbose=0).reshape(-1)
print("[PRED] min,max,mean=", float(np.nanmin(pred)), float(np.nanmax(pred)), float(np.nanmean(pred)))
print("[TARGET] min,max,mean=", float(np.nanmin(y)), float(np.nanmax(y)), float(np.nanmean(y)))
print("[STATS] nan_X=", int(np.isnan(X).sum()), " nan_pred=", int(np.isnan(pred).sum()))
# robust correlation and linear fit
mask = np.isfinite(pred) & np.isfinite(y)
# BROKEN: if mask.sum()>2:
    r = np.corrcoef(pred[mask], y[mask])[0,1]
    a,b = np.polyfit(pred[mask], y[mask], 1)
    print("[FIT] corr=", float(r), " slope=", float(a), " intercept=", float(b))
# BROKEN: else:
    print("[FIT] insufficient valid pairs")
