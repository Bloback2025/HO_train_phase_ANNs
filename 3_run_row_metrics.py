# 3_run_row_metrics.py
import pandas as pd, numpy as np
preds = pd.read_csv(r"runlogs\preds_variant2_seed20251032_20251030_205652.csv")
val = pd.read_csv(r"ho_poc_outputs\val_outputs.csv")
n = min(len(preds), len(val))
pvals = pd.to_numeric(preds['y_pred'].iloc[:n], errors='coerce')
tvals = pd.to_numeric(val['true_dclose'].iloc[:n], errors='coerce')
mask = pvals.notna() & tvals.notna()
res = pvals[mask] - tvals[mask]
mae = res.abs().mean() if not res.empty else float('nan')
rmse = (res**2).mean()**0.5 if not res.empty else float('nan')
print(f"rows_compared {mask.sum()}")
print(f"MAE {mae:.6f}")
print(f"RMSE {rmse:.6f}")
