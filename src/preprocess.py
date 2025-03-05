import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data

df = pd.read_csv("data/raw/diabetes.csv")

# Print the first 5 rows of the dataframe
print(df.head())

# print the information of columns
# print(df.info())

# Check for duplicate rows
# print("Number of duplicate rows:", df.duplicated().sum())

# check missing values
print("Missing values in each column:")
# print(df.isnull().sum())


# Scale features
scaler = StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Save preprocessed data
df.to_csv("data/processed/diabetes_preprocessed.csv", index=False)

print("✅ Data preprocessing complete. Preprocessed dataset saved.")
