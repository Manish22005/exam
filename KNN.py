import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'Review': [
        'I love this movie',
        'This film is amazing',
        'I hate this movie',
        'Worst movie ever',
        'Fantastic acting',
        'Bad story',
        'Excellent film',
        'Terrible experience'
    ],

    'Sentiment': [
        'Positive',
        'Positive',
        'Negative',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative'
    ]
}

df = pd.DataFrame(data)

# Features and Target
X = df['Review']
y = df['Sentiment']

# Convert text into numeric form
cv = CountVectorizer()

X = cv.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Test New Sentence
sample = ["Amazing movie with good acting"]

sample = cv.transform(sample)

result = model.predict(sample)

print("Prediction:", result[0])