import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

from src.utils import plot_dataset

# Download some data
california = fetch_california_housing()
print(california.data.shape)  # Features (rows, columns)
print(california.feature_names)  # List of feature names
print(california.target.shape)  # Target (housing prices)

# Convert to DataFrame
california_df = pd.DataFrame(data=california.data, columns=california.feature_names)
california_df['Target'] = california.target  # Adding the target column for housing prices

# Display the first few rows of the DataFrame
print(california_df.head())

# Plot the data
sns.set_theme(style="ticks", font_scale=0.9)
sns.pairplot(california_df, hue="HouseAge", height=1.2)

# Download some data
x = california.data[:, 0:1]
y = california.target  # Housing prices
print("X", x)
print("Y", y)

# Plot the generated data
plot_dataset(x, y)

# Apply polynomial transformation
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

print(x_poly)

# Fit the linear model on polynomial features
model = LinearRegression()
model.fit(x_poly, y)

# Sort the data for smooth plotting
x_sorted = np.sort(x, axis=0)
x_poly_sorted = poly.transform(x_sorted)

# Predictions for the sorted data
y_predict_sorted = model.predict(x_poly_sorted)

# Plotting the results
plt.scatter(x, y, color='blue')
plt.plot(x_sorted, y_predict_sorted, color='red', linewidth=2)  # Single red line
plt.title("Polynomial Regression Fit (Degree 3)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Evaluating the model
mse = mean_squared_error(y, model.predict(x_poly))
print(f"Mean Squared Error: {mse}")
