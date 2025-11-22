import json, pandas as pd
m = json.load(open(r".\manifest_2bANN.2HO.json"))
df = pd.read_csv(r".\hoxnc_testing_with_target_lagged.csv")
df_trim = df.dropna(subset=m["features"])
df_trim.to_csv(r".\hoxnc_testing_with_target_lagged_trimmed.csv", index=False)
print("[OK] wrote hoxnc_testing_with_target_lagged_trimmed.csv rows=", len(df_trim))
