from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predict
y_pred = model.predict(X_shifted).flatten()

# Metrics
mse = mean_squared_error(y_shifted, y_pred)
mae = mean_absolute_error(y_shifted, y_pred)
r2 = r2_score(y_shifted, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
