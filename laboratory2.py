import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
train_data = pd.read_csv('Obesity.csv')

# Display first few rows
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Remove duplicated
train_data = train_data.drop_duplicates()

# Encode categorical target column
if train_data['NObeyesdad'].dtype == 'object':
    le = LabelEncoder()
    train_data['NObeyesdad'] = le.fit_transform(train_data['NObeyesdad'])

# Convert categorical features to numerical
train_data = pd.get_dummies(train_data)

# Ensure target column is present
print("Columns in dataset:", train_data.columns)

# Split features and target variable
X = train_data.drop(columns=['NObeyesdad'])
y = train_data['NObeyesdad']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Choose classification type
classification_type = "binary"

if classification_type == "binary":
    model = RandomForestClassifier()
elif classification_type == "multiclass":
    model = SVC(kernel='linear')
elif classification_type == "multilabel":
    model = KNeighborsClassifier()
elif classification_type == "ordinal":
    model = LogisticRegression()
else:
    raise ValueError("Invalid classification type. Choose from 'binary', 'multiclass', 'multilabel', or 'ordinal'.")

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


