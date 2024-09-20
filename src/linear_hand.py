import numpy as np
from matplotlib import pyplot as plt

# ---------- Calculation ----------------
X = np.array([2, 3, 5, 7, 8])
Y = np.array([50, 55, 70, 80, 85])
X_mean = np.mean(X)
Y_mean = np.mean(Y)
# numerator / denominator
beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
beta_0 = Y_mean - beta_1 * X_mean
Y_predict = beta_0 + beta_1 * X
# Mean squared error (MSE)
r_squared = 1 - (np.sum((Y - Y_predict.flatten()) ** 2) / np.sum((Y - Y_mean) ** 2))
print(f"X: {X}")
print(f"Y: {Y}")
print(f"beta_1: {beta_1}")
print(f"beta_0: {beta_0}")
print(f"Y_predict: {Y_predict}")
print(f"R-squared: {r_squared:.4f}")
