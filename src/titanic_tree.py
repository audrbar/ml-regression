import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
import pandas as pd
import os
import matplotlib.pyplot as plt

path = os.getcwd()
print(path)
# --------------- DataFrame ----------------
pd.options.display.max_columns = None
# pd.options.display.max_rows = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/titanic.csv')
# Add target
df['sex_int'] = np.where(df['Sex'] == 'male', 0, 1)
print(df.head())
print(df.info())
print(df.describe())
# ---------------- Splitting the Data -------------------
X = df[['Fare', 'Age', 'SibSp', 'Pclass', 'Parch', 'sex_int']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Default criterion 'gini'
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)

print(cm)
print(accuracy)

plt.figure(figsize=(10, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=['F', 'P'])
plt.show()
