# # ============================================
# # ABALONE DATASET IMPLEMENTATION
# # a) Predict number of rings
# # b) Predict age using Linear Regression
# # ============================================

# import pandas as pd
# import numpy as np

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     accuracy_score,
#     classification_report
# )

# # ============================================
# # LOAD DATASET
# # ============================================

# column_names = [
#     'Sex', 'Length', 'Diameter', 'Height',
#     'Whole_weight', 'Shucked_weight',
#     'Viscera_weight', 'Shell_weight', 'Rings'
# ]

# data = pd.read_csv("abalone.csv", names=column_names)

# print("\nFirst 5 Rows:")
# print(data.head())

# # ============================================
# # PREPROCESSING
# # ============================================

# # Convert categorical data into numeric
# encoder = LabelEncoder()
# data['Sex'] = encoder.fit_transform(data['Sex'])

# # ============================================
# # PART (A)
# # Predict Number of Rings
# # ============================================

# print("\n==============================")
# print("PART A : RING PREDICTION")
# print("==============================")

# X = data.drop('Rings', axis=1)
# y = data['Rings']

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Classification Model
# classifier = DecisionTreeClassifier(random_state=42)

# # Train model
# classifier.fit(X_train, y_train)

# # Prediction
# y_pred = classifier.predict(X_test)

# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)

# print("\nClassification Accuracy:")
# print(accuracy)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # ============================================
# # PART (B)
# # Predict Age using Linear Regression
# # ============================================

# print("\n==============================")
# print("PART B : AGE PREDICTION")
# print("==============================")

# # Age calculation
# data['Age'] = data['Rings'] + 1.5

# X_age = data.drop(['Rings', 'Age'], axis=1)
# y_age = data['Age']

# # Split data
# X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(
#     X_age, y_age, test_size=0.2, random_state=42
# )

# # Linear Regression Model
# model = LinearRegression()

# # Train model
# model.fit(X_train_age, y_train_age)

# # Predict
# age_pred = model.predict(X_test_age)

# # Evaluation
# mse = mean_squared_error(y_test_age, age_pred)
# mae = mean_absolute_error(y_test_age, age_pred)

# print("\nMean Squared Error:")
# print(mse)

# print("\nMean Absolute Error:")
# print(mae)

# # Display some predictions
# print("\nSample Predictions:")
# results = pd.DataFrame({
#     'Actual Age': y_test_age.values[:10],
#     'Predicted Age': age_pred[:10]
# })

# print(results)





import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Dataset
cols = ['Sex','Length','Diameter','Height',
        'Whole_weight','Shucked_weight',
        'Viscera_weight','Shell_weight','Rings']

data = pd.read_csv("abalone.csv", names=cols)

# Encode Sex column
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# =========================
# PART A : Ring Prediction
# =========================

X = data.drop('Rings', axis=1)
y = data['Rings']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model1 = DecisionTreeClassifier()

model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred1))

# =========================
# PART B : Age Prediction
# =========================

data['Age'] = data['Rings'] + 1.5

X2 = data.drop(['Rings','Age'], axis=1)
y2 = data['Age']

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = LinearRegression()

model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred2))
print("MAE:", mean_absolute_error(y_test, pred2))