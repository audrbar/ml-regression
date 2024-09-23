import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = {
    'learning_duration': [10, 5, 12, 8, 15, 4, 6, 11, 7, 9, 14, 3, 2, 13, 8, 10],
    'mean_score': [6.5, 4.0, 7.5, 6.0, 8.0, 3.5, 5.5, 7.0, 5.0, 6.5, 8.5, 2.0, 4.5, 7.5, 6.0, 7.0],
    'attendance': [80, 60, 90, 70, 95, 50, 65, 85, 75, 80, 92, 40, 30, 88, 70, 80],
    'is_passed': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['learning_duration', 'mean_score', 'attendance']]
y = df['is_passed'].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=500)
model.fit(X_train_scaled, y_train)

# Set custom threshold
threshold = 0.6

# Predictions and probabilities
y_predict = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# Custom threshold prediction
y_custom_th = []
for i in y_probability:
    if i >= threshold:
        y_custom_th.append(1)
    else:
        y_custom_th.append(0)

# Outputs
print(f"Predicted classes: {y_predict}")
print(f"Predicted probabilities: {y_probability}")
print(f"Custom threshold predictions: {y_custom_th}")

# Accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_custom_th)
cm = confusion_matrix(y_test, y_custom_th)

print(f"Accuracy with custom threshold: {accuracy:.4f}")
print(f"Confusion Matrix:\n{cm}")
