from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

path = os.getcwd()
print(path)
# --------------- DataFrame ----------------
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/diabetes.csv')

# Display the first few rows of the DataFrame
print(f"DF Head: \n{df.head()}")
print(f"DF Info: \n{df.info()}")
print(f"DF Describe: \n{df.describe()}")
print(f"DF Columns: \n{df.columns}")


# Features and Target
X = df[['BloodPressure', 'Insulin', 'BMI', 'Age',]]
y = df['Outcome']

print("X", X[:5])
print("Y", y[:5])

# Split dataset to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier. Default criterion 'gini', next one 'entropy'
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)

print("Confusion_matrix: \n", cm)
print("Accuracy: \n", accuracy)

plt.figure(figsize=(12, 8))
tree.plot_tree(model, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Classifier")
plt.show()
