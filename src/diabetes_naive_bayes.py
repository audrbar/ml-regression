import os

from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

path = os.getcwd()
print(path)
# --------------- DataFrame ----------------
pd.options.display.max_columns = None
df = pd.read_csv('/Users/audrius/Documents/VCSPython/ml-regression/data/diabetes.csv')
df = df.dropna()

# Display the first few rows of the DataFrame
# print(f"DF Head: \n{df.head()}")
print(f"\n DF Info: {df.info()}")
print(f"\nDF Describe: \n{df.describe()}")
print(f"\nDF Columns: \n{df.columns}")

# Define Features and Target
X, y = (df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',]],
        df['Outcome'])

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of classifiers to iterate over
classifiers = [GaussianNB(), BernoulliNB(), MultinomialNB()]

# Store accuracy results for all classifiers
accuracy_results = []

# List of classifiers to iterate over
for classifier in classifiers:
    # Fit the model
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_predict = classifier.predict(X_test)

    # Evaluate the model
    confusion = confusion_matrix(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    classificationReport = classification_report(
        y_test, y_predict,
        target_names=['No Diabetes', 'Diabetes'],
        zero_division=1
    )
    # Append the classifier name and accuracy to the results list
    accuracy_results.append({
        'Classifier': classifier.__class__.__name__,
        'Accuracy': f"{accuracy:.4f}"
    })
    print(f"\n------------ {classifier.__class__.__name__} Evaluation -------------")
    print(f"Confusion Matrix: \n{confusion}")
    print(f"Gaussian Naive Bayes Accuracy: {accuracy:.4f}")
    print(f"Classification Report: \n{classificationReport}")

# Convert accuracy results to a DataFrame for tabular display
accuracy_df = pd.DataFrame(accuracy_results)

# Print the accuracy table
print("Accuracy Table for All Classifiers:")
print(accuracy_df)

# Set the 'Classifier' column as the index to plot directly
accuracy_df['Accuracy'] = pd.to_numeric(accuracy_df['Accuracy'], errors='coerce')

# Set the 'Classifier' column as the index to plot directly
accuracy_df.set_index('Classifier', inplace=True)

# Plot a Pandas bar chart
accuracy_df['Accuracy'].plot(kind='bar', figsize=(8, 6), color='skyblue', legend=False)

# Add titles and labels
plt.title('Accuracy of Different Naive Bayes Classifiers')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1 to represent accuracy as a percentage
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
