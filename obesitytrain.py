import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from lightgbm import LGBMClassifier

# Load the data
train_data = pd.read_csv('Obesity - Raw data.csv')


# Display the first few rows of the DataFrame
train_data.head(10)
print(train_data.head(10))

# Check for missing values
train_data.isnull().sum()
print("\nCheck Missing Values:")
print(train_data.isnull().sum())

# Check for duplicated values

duplicates = train_data.duplicated()

# Count the number of duplicated rows
num_duplicates = duplicates.sum()

print("Number of duplicated rows:", num_duplicates)

# Remove Duplicated rows
train_data = train_data.drop_duplicates()
