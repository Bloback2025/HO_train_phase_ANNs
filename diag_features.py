import json, pandas as pd
m = json.load(open(r".\manifest_2bANN.2HO.json"))
features = m["features"]
df = pd.read_csv(r".\hoxnc_full.csv")
print("[FEATURES] manifest_len=", len(features), " df_has_all=", set(features).issubset(df.columns))
print("[ORDER] first10_manifest=", features[:10])
print("[MISMATCH] missing_in_df=", sorted(set(features)-set(df.columns)))
print("[EXTRA] df_minus_manifest_sample=", sorted(set(df.columns)-set(features))[:10])
