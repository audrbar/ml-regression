import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Load dataset and explore it
iris = load_iris()

print(f"Iris dir: \n{iris.__dir__()}")
print(f"Iris data: \n{iris.data[:5]}")
print(f"Iris target: \n{iris.target}")
print(f"Iris frame: \n{iris.frame}")
print(f"Iris target names: \n{iris.target_names}")
print(f"Iris DESCR: \n{iris.DESCR}")
print(f"Iris file name: \n{iris.filename}")
print(f"Iris data module: \n{iris.data_module}")
print(f"Iris data shape: \n{iris.data.shape}")

# Define Features and Target
X, y = iris.data, iris.target

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=3, criterion='entropy', bootstrap=True, max_samples=0.8, random_state=42,
                            max_depth=3)
rf.fit(X_train, y_train)

# Predict Test data
y_predict = rf.predict(X_test)

# Evaluate model
confusion = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')
print(f"Confusion Matrix: \n{confusion}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Feature Importances: \n{iris.feature_names} \n{rf.feature_importances_}")

# Plot results
tree_ = rf.estimators_[0]
plt.figure(figsize=(8, 8))
tree.plot_tree(tree_, feature_names=iris.feature_names, class_names=['1', '2', '3', ],
               filled=True)
plt.show()
