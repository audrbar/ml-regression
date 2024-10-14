import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import ssl
import nltk
from nltk.tokenize import word_tokenize

# Bag o Words
documents = ['coronavirus is a highly infectious disease',
             'coronavirus affects older people the most',
             'older people are at high risk due to this disease']

vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(documents)

# Bag of Words BoW
print(vectorizer.get_feature_names_out())
print(X.toarray())
print(sorted(vectorizer.vocabulary_.keys()))

# One-Hot Encoding
encoder = OneHotEncoder()
encoded_docs = encoder.fit_transform([[word] for doc in documents for word in doc.split()])

print(encoder.get_feature_names_out())
print(encoded_docs.toarray())

# Term Frequency–Inverse Document Frequency or TF-IDF
tfidf = TfidfVectorizer()
X_tf = tfidf.fit_transform(documents)

print(tfidf.get_feature_names_out())
print(X_tf.toarray())

df = pd.DataFrame(X_tf[0].T.todense(), index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)

# Natural Language Toolkit (nltk) library
print(nltk.__path__)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()
sentence = '''Good muffins cost $3.88\nin New York.  Please buy me
... two of them.\n\nThanks.'''
tokens = word_tokenize(sentence)
print(tokens)
