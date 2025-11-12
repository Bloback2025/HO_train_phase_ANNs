import pickle, numpy as np, sys, os
fn = r"ho_artifact_outputs\scaler.pkl"
if not os.path.exists(fn):
    print("MISSING", fn); sys.exit(2)
sc = pickle.load(open(fn,"rb"))
X = np.load("X_eval.npy")
Xs = sc.transform(X)
np.save("X_eval_scaled.npy", Xs)
print("SAVED X_eval_scaled.npy", X.shape, Xs.shape)
