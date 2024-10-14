import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

metrics = ['minkowski', 'cityblock', 'euclidean', 'manhattan']
neighbors = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
accuracies = []
for metric in metrics:
    for neighbor in neighbors:
        knn = KNeighborsClassifier(n_neighbors=neighbor, metric=metric)
        knn.fit(X_train, y_train)

        y_predict = knn.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_predict)
        accuracies.append((metric, neighbor, accuracy))
        print(f"Accuracy for {metric}, {neighbor}: {accuracy}")

print("Final accuracies for each metric:")
for metric, neighbor, accuracy in accuracies:
    print(f"{metric}, {neighbor}: {accuracy}")

# Plotting the results as a line chart
fig, ax = plt.subplots(figsize=(10, 6))

for metric in metrics:
    acc_by_metric = [accuracy for m, n, accuracy in accuracies if m == metric]
    ax.plot(neighbors, acc_by_metric, marker='o', label=f"{metric}")

ax.set_title('Accuracy for Different Metrics and Neighbors')
ax.set_xlabel('Number of Neighbors')
ax.set_ylabel('Accuracy')
ax.legend(title='Metric')
ax.grid(True)

plt.tight_layout()
plt.show()
