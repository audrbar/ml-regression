from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# Load the cancer dataset
cancer = load_breast_cancer()
print(f"Cancer data shape: {cancer.data.shape}")
print(f"Cancer feature names: \n{cancer.feature_names}")
print(f"cancer target shape: {cancer.target.shape}")
print(f"cancer target names: {cancer.target_names}")
print(f"cancer target: \n{cancer.target}")

# Define Features and Target
X, y = cancer.data, cancer.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the GridSearch
param_grid = {
    'metric': ['minkowski', 'cityblock', 'euclidean', 'manhattan'],
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
