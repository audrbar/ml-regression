import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Download some data
california = fetch_california_housing()
print(california.data.shape)  # Features (rows, columns)
print(california.feature_names)  # List of feature names
print(california.target.shape)  # Target (housing prices)

# Convert to DataFrame
# california_df = pd.DataFrame(data=california.data, columns=california.feature_names)
# california_df['Target'] = california.target  # Adding the target column for housing prices
#
# # Display the first few rows of the DataFrame
# print(california_df.head())
#
# # Plot the data Pair Plot
# sns.set_theme(style="ticks", font_scale=0.9)
# sns.pairplot(california_df, hue="HouseAge", height=1.2)

# Prepare full feature set and target
X_full = california.data
y = california.target

# Split the full dataset into train and test
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Store R-squared values for each feature removal case
feature_names = california.feature_names
r_squared_values = []

# Iterate over features, removing one at a time
for i in range(X_full.shape[1]):
    X_train_reduced = np.delete(X_train_full, i, axis=1)
    X_test_reduced = np.delete(X_test_full, i, axis=1)

    # Train the model on the reduced feature set
    model = LinearRegression()
    model.fit(X_train_reduced, y_train)

    # Predicting with the model
    Y_predict = model.predict(X_test_reduced)

    # Calculate R-squared value
    r_squared = model.score(X_test_reduced, y_test)
    r_squared_values.append((feature_names[i], r_squared))

    # Output the results for each feature removed
    print(f"R-squared after removing {feature_names[i]}: {r_squared:.4f}")

    # Plot the actual vs predicted housing prices
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, Y_predict, color='blue', label='Predictions vs Actual', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal Line')
    plt.xlabel('Actual Housing Prices')
    plt.ylabel('Predicted Housing Prices')
    plt.title(f'Actual vs Predicted: Removed {feature_names[i]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Sort the R-squared values for better understanding of the feature importance
r_squared_values_sorted = sorted(r_squared_values, key=lambda x: x[1], reverse=True)

# Display the sorted results
print("R-squared values after removing each feature:")
for feature, r_sq in r_squared_values_sorted:
    print(f"{feature}: {r_sq:.4f}")
