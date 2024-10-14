import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

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

# Define Features and Target
X, y = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',]],
        df['Outcome'])
print(f"X Head: \n{X.head()}")
print(f"y Head: \n{y.head()}")

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion='entropy', bootstrap=True, max_samples=0.9, random_state=42,
                            max_depth=10)
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
print(f"Feature Importance's: \n{X.columns} \n{rf.feature_importances_}")

# Plot results
if len(rf.estimators_) < 4:
    n_estimators = len(rf.estimators_)
    fig, axes = plt.subplots(1, n_estimators, figsize=(20, 8))
    for idx, (estimator, ax) in enumerate(zip(rf.estimators_, axes)):
        tree.plot_tree(estimator, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True, ax=ax)
        ax.set_title(f"Tree {idx + 1}")
    plt.tight_layout()
    plt.show()
else:
    for idx, estimator in enumerate(rf.estimators_, start=1):
        tree_ = estimator
        plt.figure(figsize=(8, 8))
        tree.plot_tree(tree_, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
        plt.title(f"Decision Tree {idx} in Random Forest")
        plt.show()
