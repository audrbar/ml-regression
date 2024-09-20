import numpy as np
from matplotlib import pyplot as plt

# ---------- Calculation ----------------
X = np.array([50, 80, 120, 160, 200])
Y = np.array([150, 220, 320, 410, 480])
X_mean = np.mean(X)
Y_mean = np.mean(Y)
# numerator / denominator
beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
beta_0 = Y_mean - beta_1 * X_mean
Y_predict = beta_0 + beta_1 * X
# Predict Price fo 150 square meters
Y_150_predict = beta_0 + beta_1 * 150
# Mean Squared Error (MSE)
r_squared = 1 - (np.sum((Y - Y_predict.flatten()) ** 2) / np.sum((Y - Y_mean) ** 2))
print(f"X: {X}")
print(f"Y: {Y}")
print(f"beta_1: {beta_1}")
print(f"beta_0: {beta_0}")
print(f"Y_predict: {Y_predict}")
print(f"Y_150_predict: {Y_150_predict}")
print(f"R-squared: {r_squared:.4f}")
# ---------- Plot ----------------
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_predict, color='red', linewidth=2, label=f'Regression line: Y = {beta_0:.2f} + {beta_1:.2f}X')
plt.xlabel('area')
plt.ylabel('price')
plt.title('Simple Linear Regression Area v.s. Price')
plt.legend()
plt.grid(True)
plt.show()
