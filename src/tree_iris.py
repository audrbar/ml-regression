from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
print(f"Iris data shape: \n{iris.data.shape}")
print(f"Iris feature names: \n{iris.feature_names}")
print(f"Iris target shape: \n{iris.target.shape}")
print(f"Iris target names: \n{iris.target_names}")
print(f"Iris target: \n{iris.target}")

# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Target'] = iris.target

# Display the first few rows of the DataFrame
print(f"DF Head: \n{iris_df.head()}")
print(f"DF Info: \n{iris_df.info()}")
print(f"DF Describe: \n{iris_df.describe()}")

# Plot the data Pair Plot
sns.set_theme(style="ticks", font_scale=0.9)
sns.pairplot(iris_df, hue='Target', height=1.8)
plt.show()

# Define Features and Target
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_df['Target']
print("X: \n", X[:5])
print("Y: \n", y[:5])

# Split dataset to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Decision Tree Classifier. Default criterion 'gini', next one 'entropy'
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)

print("Confusion_matrix: \n", cm)
print("Accuracy: \n", accuracy)

# Visualize Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Visualize Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Plot")
plt.show()

# Store Accuracy
accuracy_values = [accuracy]
print(f"Stored Accuracy: {accuracy_values}")

# Feature Importance
feature_importance = model.feature_importances_
print("Feature Importance: ", feature_importance)

# Plot Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(X.columns, feature_importance, color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
