import numpy as np, sys, os
for fn in ("X_eval.npy","y_eval.npy"):
    if not os.path.exists(fn):
        print("MISSING", fn); sys.exit(2)
X = np.load("X_eval.npy")
y = np.load("y_eval.npy")
print("LOADED", "X_eval.npy", X.shape, "y_eval.npy", y.shape)
print("X dtype", X.dtype, "y dtype", y.dtype)
print("X head (first 3 rows):")
print(X[:3].tolist())
print("y head (first 10):")
print(y[:10].tolist())
