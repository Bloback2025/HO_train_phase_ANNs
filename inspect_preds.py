import numpy as np
X = np.load("X_eval.npy")
y = np.load("y_eval.npy")
from tensorflow.keras.models import load_model
m = load_model(r"ho_artifact_outputs\2bANN2_HO_model.keras")
pred = m.predict(X).reshape(-1)
import numpy as _n
def show(n=10):
    print("X head (first row):", X[0].tolist())
    print("y head:", y[:n].tolist())
    print("pred head:", pred[:n].tolist())
    print("y mean:", float(_n.mean(y)), "pred mean:", float(_n.mean(pred)))
    print("y std:", float(_n.std(y)), "pred std:", float(_n.std(pred)))
show(10)
_n.save("nulltest_preds.npy", pred)
print("SAVED nulltest_preds.npy")
