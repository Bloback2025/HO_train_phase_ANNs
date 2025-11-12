import pandas as pd, numpy as np, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
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
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"RESULTS: test_rows={len(y_test)} MAE={mae:.6g} R2={r2:.6g}")
print("------ sample residuals (true -> pred | residual) ------")
k = min(10, len(y_test))
for i in range(k):
    print(f"{i+1:02d}: true={y_test[i]:.6g} -> pred={preds[i]:.6g} | res={preds[i]-y_test[i]:.6g}")
