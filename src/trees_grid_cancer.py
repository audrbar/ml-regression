from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the cancer dataset
cancer = load_breast_cancer()
print(f"cancer data shape: \n{cancer.data.shape}")
print(f"cancer feature names: \n{cancer.feature_names}")
print(f"cancer target shape: \n{cancer.target.shape}")
print(f"cancer target names: \n{cancer.target_names}")
print(f"cancer target: \n{cancer.target}")

# Define Features and Target
X, y = cancer.data, cancer.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [17, 18, 23],
    'max_depth': [4, 5, 6],
    'bootstrap': [True]
}

# Set up GridSearchCV
# CV - cross validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=6, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters found: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: \n{accuracy}")
print(f"Confusion Matrix: \n{confusion}")
print(f"Feature Importance's: \n{X.columns} \n{rf.feature_importances_}")
