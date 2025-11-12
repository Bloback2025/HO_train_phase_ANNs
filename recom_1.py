from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_true = y_test_s[1:]          # shift true values forward
y_pred_aligned = y_pred[:-1]   # drop last prediction to align

print("Aligned shapes:", y_true.shape, y_pred_aligned.shape)
print("MAE aligned:", mean_absolute_error(y_true, y_pred_aligned))
print("RMSE aligned:", mean_squared_error(y_true, y_pred_aligned, squared=False))
print("R2 aligned:", r2_score(y_true, y_pred_aligned))
