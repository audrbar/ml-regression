from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'learning_duration': [10, 5, 12, 8, 15, 4, 6, 11, 7, 9, 14, 3, 2, 13, 8, 10],
    'attendance': [80, 60, 90, 70, 95, 50, 65, 85, 75, 80, 92, 40, 30, 88, 70, 80],
    'is_passed': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['learning_duration', 'attendance']]
y = df['is_passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Default criterion 'gini'
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)

print(cm)
print(accuracy)

plt.figure(figsize=(10, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=['F', 'P'])
plt.show()
