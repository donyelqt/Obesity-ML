import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from lightgbm import LGBMClassifier
import joblib

# Load the data
train_data = pd.read_csv('ObesityyY - Raw data.csv')

# Display the first few rows of the DataFrame
train_data.head()

# Check for missing values
train_data.isnull().sum()

# Check for duplicated values
duplicates = train_data.duplicated()
# Count the number of duplicated rows
num_duplicates = duplicates.sum()
print("Number of duplicated rows:", num_duplicates)

# Remove duplicated
train_data = train_data.drop_duplicates()

# Save the cleaned dataset
train_data.to_csv('cleaned_obesity.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_obesity.csv'.")

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

