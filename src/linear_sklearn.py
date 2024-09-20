import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------- Calculation ----------------
X = np.array([2, 3, 5, 7, 8]).reshape(-1, 1)
Y = np.array([50, 55, 70, 80, 85])
model = LinearRegression()
model.fit(X, Y)
beta_1 = model.coef_[0]
beta_0 = model.intercept_
Y_predict = model.predict(X)
r_squared = model.score(X, Y)
print(f"X: {X}")
print(f"Y: {Y}")
print(f"beta_1: {beta_1}")
print(f"beta_0: {beta_0}")
print(f"Y_predict: {Y_predict}")
print(f"R-squared: {r_squared:.4f}")
# ---------- Plot ----------------
plt.scatter(X, Y, color='blue', alpha=0.5, label='Data points')
plt.plot(X, Y_predict, color='red', linewidth=2, label=f'Regression line: Y = {beta_0:.2f} + {beta_1:.2f}X')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
