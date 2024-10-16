from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset and explore it
iris = load_iris()

print(f"Iris dir: \n{iris.__dir__()}")
print(f"Iris data: \n{iris.data[:5]}")
print(f"Iris target: {iris.target}")
print(f"Iris frame: {iris.frame}")
print(f"Iris target names: {iris.target_names}")
print(f"Iris DESCR: {iris.DESCR}")
print(f"Iris file name: {iris.filename}")
print(f"Iris data module: {iris.data_module}")
print(f"Iris data shape: {iris.data.shape}")

# Define Features and Target
X, y = iris.data, iris.target

# Split data into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predict = gnb.predict(X_test)

confusion = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
classification_report = classification_report(y_test, y_predict, target_names=iris.target_names)
print(f"Confusion Matrix: \n{confusion}")
print(f"Gaussian Naive Bayes Accuracy: {accuracy}")
print(f"Classification Report: \n{classification_report}")
