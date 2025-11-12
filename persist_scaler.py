import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

p = Path('hoxnc_full.with_target.csv')
df = pd.read_csv(p)
n = len(df)
test_n = int(round(0.15 * n))
val_n = int(round(0.1 * (n - test_n)))
train_end = n - test_n - val_n
features = ['Open','High','Low','Close']
X_train = df[features].astype(float).values[:train_end]
scaler = StandardScaler().fit(X_train)
outdir = Path('models')
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('WROTE: models/scaler.pkl rows_trained=', len(X_train))
