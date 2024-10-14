import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

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
X, y = (df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',]],
        df['Outcome'])
print(f"X Head: \n{X.head()}")
print(f"y Head: \n{y.head()}")

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Set Params
param_grid = {
    'n_estimators': [35, 40, 45],
    'max_depth': [4, 5, 6]
}

# Set up GridSearchCV (Cross Validation)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_predi = best_rf.predict(X_test)

confusion = confusion_matrix(y_test, y_predi)
accuracy = accuracy_score(y_test, y_predi)

print(f"Accuracy: \n{accuracy}")
print(f"Confusion Matrix: \n{confusion}")
print(f"Feature Importance's: \n{X.columns} \n{rf.feature_importances_}")
