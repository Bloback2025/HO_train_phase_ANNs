import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y = np.load("y_eval.npy")
pred = np.load("nulltest_preds.npy")
print("MSE:", mean_squared_error(y,pred))
print("MAE:", mean_absolute_error(y,pred))
print("R2:", r2_score(y,pred))
