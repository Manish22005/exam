# Sentiment Analysis using Naive Bayes

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training Data
documents = [
    "I love this product",
    "This is an amazing movie",
    "I feel very happy today",
    "This is a great experience",
    "I hate this product",
    "This is a terrible movie",
    "I feel very sad today",
    "This is a bad experience"
]

# Labels
labels = [
    "Positive",
    "Positive",
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Negative"
]

# Convert Text into Numerical Form
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)

# Train Naive Bayes Classifier
model = MultinomialNB()

model.fit(X, labels)

# Test Document
test_document = ["This product is bad"]

# Convert Test Document
test_data = vectorizer.transform(test_document)

# Predict Sentiment
prediction = model.predict(test_data)

# Output
print("Document:", test_document[0])
print("Predicted Sentiment:", prediction[0])