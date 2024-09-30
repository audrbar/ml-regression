import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Get data
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/cars.csv')
print(df.info())
print(df.head())

# Add target
df['going_to_sel'] = np.where(df['owner'] == 'First Owner', 0, 1)

# Assign features and target
X = df[['year', 'selling_price', 'km_driven']]
y = df['going_to_sel'].to_numpy()
print(X)
print(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(X_train_scaled, y_train)

# Predictions and probabilities
y_predict = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# Set custom threshold
thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]

# Store accuracy for each threshold
accuracies = []

# Iterate over different thresholds
for threshold in thresholds:
    # Generate predictions based on the current threshold
    y_custom_th = [1 if prob >= threshold else 0 for prob in y_probability]

    # Calculate accuracy for the current threshold
    accuracy = accuracy_score(y_test, y_custom_th)
    accuracies.append(accuracy)

    # Print accuracy for each threshold
    print(f"Threshold: {threshold}, Accuracy: {accuracy:.4f}")

# Plot the accuracies for each threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Test best threshold
threshold = 0.4
y_custom_th = []
for i in y_probability:
    if i >= threshold:
        y_custom_th.append(1)
    else:
        y_custom_th.append(0)

# Accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_custom_th)
cm = confusion_matrix(y_test, y_custom_th)

print(f"\nAccuracy with best threshold: {accuracy:.4f}")
print(f"Confusion Matrix with best threshold:\n{cm}")
