# Titanic Dataset Preprocessing

# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

# Display First 5 Rows
print("First 5 Rows:\n")
print(df.head())
print(df.tail())

# ---------------------------------------
# Handling Missing Values
# ---------------------------------------

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because many values are missing
df.drop('Cabin', axis=1, inplace=True)

# ---------------------------------------
# Convert Categorical Data to Numerical
# ---------------------------------------

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])

df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# ---------------------------------------
# Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# ---------------------------------------
# Display Preprocessed Dataset
# ---------------------------------------

print("\nPreprocessed Dataset:\n")
print(df.head())