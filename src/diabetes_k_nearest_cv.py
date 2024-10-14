import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score

path = os.getcwd()
print(path)
# --------------- DataFrame ----------------
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/diabetes.csv')
df = df.dropna()

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

# Define the parameter grid for the GridSearch
param_grid = {
    'metric': ['minkowski', 'cityblock', 'euclidean', 'manhattan'],
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier()

# Set up GridSearchCV (Cross Validation)
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid,
                           cv=6, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Find and Print the best hyperparameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Evaluate the best params
knn_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
knn_best.fit(X_train, y_train)

y_predict = knn_best.predict(X_test)

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
