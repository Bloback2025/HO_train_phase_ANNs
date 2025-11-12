import numpy as np, pandas as pd, sys, os
src = "hoxnc_testing.csv"
if not os.path.exists(src):
    print("ERROR: source not found", src); sys.exit(2)
df = pd.read_csv(src)
required = ["Open","High","Low","Close"]
for c in required:
    if c not in df.columns:
        print("ERROR: missing column", c); sys.exit(2)
# Prefer explicit Close_t+1 target column if present
if "Close_t+1" in df.columns:
    y = df["Close_t+1"].values
else:
    # infer next-day Close from shifted Close
    y = df["Close"].shift(-1).values
X_full = df[required].values
if "Close_t+1" not in df.columns:
    X = X_full[:-1]
else:
    X = X_full
# Align lengths
n = min(len(X), len(y))
X = X[:n]
y = y[:n]
if X.shape[1] != 4:
    print("ERROR: built X does not have 4 features", X.shape); sys.exit(3)
np.save("X_eval.npy", X)
np.save("y_eval.npy", y)
print("SAVED X_eval.npy y_eval.npy", X.shape, y.shape)
