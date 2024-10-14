import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score

# Load dataset and explore it
iris = load_iris()

print(f"Iris dir: \n{iris.__dir__()}")
print(f"Iris data: \n{iris.data[:5]}")
print(f"Iris target: {iris.target}")
print(f"Iris frame: {iris.frame}")
print(f"Iris target names: {iris.target_names}")
print(f"Iris DESCR: {iris.DESCR}")
print(f"Iris file name: {iris.filename}")
print(f"Iris data module: {iris.data_module}")
print(f"Iris data shape: {iris.data.shape}")

# Define Features and Target
X, y = iris.data, iris.target

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

metrics = ['minkowski', 'cityblock', 'euclidean', 'manhattan']
accuracies = []
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    knn.fit(X_train, y_train)

    y_predict = knn.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_predict)
    accuracies.append((metric, accuracy))
    print(f"Accuracy for {metric}: {accuracy}")

# confusion = confusion_matrix(y_test, y_predict)
# precision = precision_score(y_test, y_predict, average='macro')
# recall = recall_score(y_test, y_predict, average='macro')
# f1 = f1_score(y_test, y_predict, average='macro')
# print(f"\nConfusion Matrix: \n{confusion}")
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
#
# print(X_test[:, 0], X_test[:, 1])
#
# plt.figure(figsize=(10, 10))
# plt.scatter(X_test[:, 0], X_test[:, 1], cmap='coolwarm', c=y_predict, marker='o', label='Predicted')
# plt.scatter(X_test[:, 0], X_test[:, 1], cmap='coolwarm', c=y_predict, marker='x', label='True')
# plt.show()

# Print overall accuracies
print("Final accuracies for each metric:")
for metric, accuracy in accuracies:
    print(f"{metric}: {accuracy}")
