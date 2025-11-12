import pandas as pd, numpy as np, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pathlib import Path, PurePath
p = Path('hoxnc_full.with_target.csv')
df = pd.read_csv(p)
features = ['Open','High','Low','Close']
target = 'Close_t+1'
n = len(df)
test_n = int(round(0.15 * n))
val_n = int(round(0.1 * (n - test_n)))
train_end = n - test_n - val_n
val_end = train_end + val_n
X = df[features].astype(float).values
y = df[target].astype(float).values
X_train = X[:train_end]; X_test = X[val_end:]; y_test = y[val_end:]
scaler = StandardScaler().fit(X_train)
X_test_s = scaler.transform(X_test)
model = tf.keras.models.load_model(Path('models/ohlc_best.keras').resolve())
preds = model.predict(X_test_s).reshape(-1)
k = min(10, len(X_test))
print("INFO: rows_total=", n, "train_end=", train_end, "val_end=", val_end, "test_rows=", len(X_test))
print("INFO: model_input_dim=", X_train.shape[1])
print("------ first", k, "test rows (features | true -> pred) ------")
for i in range(k):
    feats = ", ".join(f"{v:.6g}" for v in X_test[i])
    print(f"{i+1:02d}: [{feats}] | true={y_test[i]:.6g} -> pred={preds[i]:.6g}")
import statistics
print("------ summary stats ------")
print("test_true min/mean/max:", float(min(y_test)), float(statistics.mean(y_test)), float(max(y_test)))
print("pred min/mean/max:", float(min(preds)), float(statistics.mean(preds)), float(max(preds)))
print("scaler mean:", ", ".join(f"{m:.6g}" for m in scaler.mean_))
print("scaler scale:", ", ".join(f"{s:.6g}" for s in scaler.scale_))
