import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# Load the dataset
newsgroups = fetch_20newsgroups()

# Explore the dataset
print(f"\nNumber of documents: {len(newsgroups.data)}")
print(f"\nFirst document: \n{newsgroups.data[0]}")
print(f"Number of categories: {len(newsgroups.target_names)}")
print(f"Categories: {newsgroups.target_names}")
print(f"\nTarget (Category index) for the first document: {newsgroups.target[0]}")
print(f"Target (Category name) for the first document: {newsgroups.target_names[newsgroups.target[0]]}\n")

# Extract the data (text) and target (categories)
X = newsgroups.data  # The text documents
y = newsgroups.target  # The target categories
print(f"X: \n{X[0]}")
print(f"y: {y[0]}")

# Preprocess the text data using TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)  # Limit to 10,000 features for efficiency
X_tfidf = tfidf.fit_transform(X)  # Transform the text data into a TF-IDF matrix

print(f"\nBagofWords Length: {len(tfidf.get_feature_names_out())}")
print(f"X_tfidf Length: {len(X_tfidf.toarray())}")
print(f"\nBagofWords[5350:5360]: \n{tfidf.get_feature_names_out()[5350:5360]}")
print(f"X_tfidf[500:515]: \n{X_tfidf.toarray()[500:515]}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier on the TF-IDF data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the test set and calculate accuracy
y_predict = model.predict(X_test)

# Evaluate model
confusion = confusion_matrix(y_test, y_predict)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='macro')
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')
print(f"Confusion Matrix: \n{confusion}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

df = pd.DataFrame(X_tfidf[0].T.todense(), index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(f"\ndf length: {len(df)}")
print(f"\ndf info: {df.info()}")
print(f"\nImportant words: {len(df[df['TF-IDF'] > 0])}")
print(f"Important words: \n{df[df['TF-IDF'] > 0]}")
