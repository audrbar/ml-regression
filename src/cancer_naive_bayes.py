import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the cancer dataset
cancer = load_breast_cancer()
print(f"Cancer data shape: {cancer.data.shape}")
print(f"Cancer feature names: \n{cancer.feature_names}")
print(f"cancer target shape: {cancer.target.shape}")
print(f"cancer target names: {cancer.target_names}")
print(f"cancer target: \n{cancer.target}")

# Define Features and Target
X, y = cancer.data, cancer.target

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
        target_names=cancer.target_names,
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
accuracy_naive_bayes = accuracy_results[1]
# Print the accuracy table
print("Accuracy Table for All Classifiers:")
print(accuracy_df)

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
