import json, pandas as pd, numpy as np
m = json.load(open(r".\manifest_2bANN.2HO.json"))
target = m["target"]
df = pd.read_csv(r".\hoxnc_full.csv")
y = df[target].astype(float)
print("[TARGET] name=", target, " nonnull=", int(y.notnull().sum()))
print("[TARGET] stats min=", float(np.nanmin(y)), " max=", float(np.nanmax(y)), " mean=", float(np.nanmean(y)))
