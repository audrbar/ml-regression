from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print(diabetes.data.shape)
print(diabetes.feature_names)
print(diabetes.target.shape)

# Select the first feature for X and the target for y
X_full = diabetes.data  # Selecting the 'age' feature [:, 0:1]
y = diabetes.target  # Diabetes cases
print("X", X_full[:5])
print("Y", y[:5])

# Split dataset to train and test data
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# Store R-squared and feature importance for each model
feature_names = diabetes.feature_names
r_squared_values_unscaled = []
r_squared_values_scaled = []

# Iterate over features, removing one at a time (unscaled data)
for i in range(X_full.shape[1]):
    # Select only the current feature for training
    X_train_un = X_train_full[:, [i]]
    X_test_un = X_test_full[:, [i]]

    # Train the model on the reduced feature set
    model = LinearRegression()
    model.fit(X_train_un, y_train)

    # Predicting with the model
    Y_predict_un = model.predict(X_test_un)

    # Calculate R-squared value
    r_squared = model.score(X_test_un, y_test)
    r_squared_values_unscaled.append((feature_names[i], r_squared))

    # Output the results for each feature removed
    print(f"R-squared for {feature_names[i]}: {r_squared:.4f}")

    # Plot the actual vs predicted housing prices
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test_un, y_test, color='blue', alpha=0.5, label='Data points')
    plt.plot(X_test_un, Y_predict_un, color='red', linewidth=2, label=f'Regression line')
    plt.xlabel('Diabetes Cases')
    plt.ylabel(f'{feature_names[i]}')
    plt.title(f'Simple Linear Regression: Feature {feature_names[i]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Iterate over features, removing one at a time (scaled data)
for i in range(X_full.shape[1]):
    # Select only the current feature for training
    X_train_sc = X_train_scaled[:, [i]]
    X_test_sc = X_test_scaled[:, [i]]

    # Train the model on the reduced feature set
    model = LinearRegression()
    model.fit(X_train_sc, y_train)

    # Predicting with the model
    Y_predict_sc = model.predict(X_test_sc)

    # Calculate R-squared value
    r_squared_sc = model.score(X_test_sc, y_test)
    r_squared_values_scaled.append((feature_names[i], r_squared_sc))

    # Output the results for each feature removed
    print(f"R-squared for {feature_names[i]}: {r_squared_sc:.4f}")

    # Plot the actual vs predicted housing prices
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test_sc, y_test, color='blue', alpha=0.5, label='Data points')
    plt.plot(X_test_sc, Y_predict_sc, color='red', linewidth=2, label=f'Regression line')
    plt.xlabel('Diabetes Cases')
    plt.ylabel(f'{feature_names[i]}')
    plt.title(f'Simple Linear Regression: Feature {feature_names[i]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display the sorted results for unscaled and scaled data
r_squared_values_unscaled_sorted = sorted(r_squared_values_unscaled, key=lambda x: x[1], reverse=True)
r_squared_values_scaled_sorted = sorted(r_squared_values_scaled, key=lambda x: x[1], reverse=True)

print("\nR-squared values using only each feature (Unscaled):")
for feature, r_sq in r_squared_values_unscaled_sorted:
    print(f"{feature}: {r_sq:.4f}")

print("\nR-squared values using only each feature (Scaled):")
for feature, r_sq in r_squared_values_scaled_sorted:
    print(f"{feature}: {r_sq:.4f}")
