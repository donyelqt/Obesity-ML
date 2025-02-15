import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('Obesity - Raw data.csv')

# Step 1: Data Cleaning

## 1.1 Handle Missing Values
# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values before cleaning:\n{missing_values}")

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Drop rows where the target variable is missing
df = df.dropna(subset=['NObeyesdad'])

## 1.2 Handle Outliers
# Visualize numerical features to detect outliers
for col in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Apply Z-score to detect and remove outliers
df[numerical_features] = df[numerical_features].apply(zscore)
df = df[(np.abs(df[numerical_features]) < 3).all(axis=1)]  # Remove rows with Z-score > 3

# Step 2: Data Transformation

## 2.1 Normalize Numerical Features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

## 2.2 Encode Categorical Features
# Label encode binary categorical variables
binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC']
le = LabelEncoder()
df[binary_cols] = df[binary_cols].apply(le.fit_transform)

# One-hot encode non-ordinal categorical variables
df = pd.get_dummies(df, columns=['MTRANS'], drop_first=True)

## 2.3 Feature Engineering
# Create new feature 'AgeGroup'
# df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'Adult', 'Middle-Aged', 'Senior'])
# df['AgeGroup'] = df['AgeGroup'].astype(str)

# Encode AgeGroup
# df['AgeGroup'] = le.fit_transform(df['AgeGroup'])

# Step 3: Data Reduction Methods

## 3.1 Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
df_pca = pca.fit_transform(df[numerical_features])
df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

# Add PCA components to the dataset
df = pd.concat([df, df_pca], axis=1)

## 3.2 Feature Selection using RFE
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
model = RandomForestClassifier(n_estimators=100, random_state=42)
print(f"Missing values in y: {y.isnull().sum()}")

rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

# Keep only the selected features
df = df[selected_features.to_list() + ['NObeyesdad']]

# Step 4: Data Splitting
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the cleaned and transformed dataset
df_cleaned = pd.concat([X, y], axis=1)
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Data preprocessing completed successfully! The cleaned dataset is saved as 'cleaned_dataset.csv'.")
