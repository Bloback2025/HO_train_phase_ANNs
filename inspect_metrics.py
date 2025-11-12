import pandas as pd, numpy as np, os
p = r"C:\Users\loweb\AI_Financial_Sims\HO\HO 1st time 5080\ho_poc_outputs\metrics.csv"
df = pd.read_csv(p)
print("rows,cols:", df.shape)
print(df.describe().T[['count','mean','std','min','max']].head(20))
print("\nColumns:", df.columns.tolist())
# detect model vs persistence candidate columns
c_model = next((c for c in df.columns if 'fish' in c.lower() or 'model' in c.lower()), None)
c_persist = next((c for c in df.columns if 'naive' in c.lower() or 'persist' in c.lower()), None)
print("detected columns:", c_model, c_persist)
# BROKEN: if c_model and c_persist:
    m = df[c_model].dropna().astype(float)
    p = df[c_persist].dropna().astype(float)
    print("model mean,std,min,max:", m.mean(), m.std(), m.min(), m.max())
    print("persistence mean,std,min,max:", p.mean(), p.std(), p.min(), p.max())
    print("corr(model,persistence):", m.corr(p))
