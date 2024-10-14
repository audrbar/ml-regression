import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/titanic.csv')

# The features and target
X = df[['Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Embarked']]
y = df['Survived']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor for column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('name_tfidf', TfidfVectorizer(), 'Name'),
        ('ticket_tfidf', TfidfVectorizer(), 'Ticket'),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Sex', 'Embarked'])
    ]
)

# Create the final pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing step
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Classifier
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_predict = pipeline.predict(X_test)

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
