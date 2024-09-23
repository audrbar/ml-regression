import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = {
    'age': [25, 40, 35, 50, 23, 45, 30, 31, 34, 40, 28, 37, 24, 29, 55, 42],
    'income': [30000, 50000, 45000, 60000, 28000, 58000, 35000, 36000, 40000, 52000,
               33000, 49000, 31000, 42000, 70000, 48000],
    'sex': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    'time_on': [30, 80, 55, 120, 20, 90, 45, 50, 40, 100, 35, 70, 25, 60, 130, 75],
    'bought': [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['age', 'income', 'sex', 'time_on']]
y = df['bought'].to_numpy()

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
thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9998]

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
