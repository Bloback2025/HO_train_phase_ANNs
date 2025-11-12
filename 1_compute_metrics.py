# 1_compute_metrics.py
import pandas as pd
import numpy as np
p = r"runlogs\preds_variant2_seed20251032_20251030_205652.csv"
df = pd.read_csv(p)
res = df['y_pred'].astype(float) - df['y_true'].astype(float)
mae = np.mean(np.abs(res))
rmse = np.sqrt(np.mean(res**2))
print(f"file={p}")
print(f"rows={len(df)}")
print(f"MAE={mae:.6f}")
print(f"RMSE={rmse:.6f}")
