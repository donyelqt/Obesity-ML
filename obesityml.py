import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib

# Load dataset
train_data = pd.read_csv('assets/Obesity.csv')

# Display first few rows
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Remove duplicates
train_data = train_data.drop_duplicates()

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Handle outliers using Z-score
z_scores = np.abs(stats.zscore(train_data[numerical_features]))
outliers = (z_scores > 3).any(axis=1)  # Changed to .any() to remove rows with any outlier
train_data = train_data[~outliers]

# Encode target variable into numerical labels
label_encoder = LabelEncoder()
train_data['NObeyesdad'] = label_encoder.fit_transform(train_data['NObeyesdad'])

# Note: We’re using pd.get_dummies later, so preprocessing categorical features here isn’t needed yet

# Split features and target variable
X = train_data.drop(columns=['NObeyesdad'])
y = train_data['NObeyesdad']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define data preprocessing steps
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Choose classification type
classification_type = "lightgbm"  # Changed it with the model below

# Select and initialize the model
if classification_type == "rfc":
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif classification_type == "svc":
    model = SVC(random_state=42)
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
elif classification_type == "kneighbors":
    model = KNeighborsClassifier()
    param_grid = {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance']
    }
elif classification_type == "logistic":
    model = LogisticRegression(random_state=42, max_iter=1000)
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']  # liblinear supports l1 and l2
    }
elif classification_type == "lightgbm":
    model = LGBMClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [3, 5, 7, 9, 11],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'classifier__subsample': [0.5, 0.7, 0.9, 1.0],
    'classifier__colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'classifier__min_child_samples': [10, 20, 30, 40, 50],
    'classifier__reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'classifier__reg_lambda': [0.0, 0.1, 0.5, 1.0],
    'classifier__min_child_weight': [1e-3, 1e-2, 0.1, 1, 10]
    }
else:
    raise ValueError("Invalid classification type. Choose from 'rfc', 'svc', 'kneighbors', 'logistic', or 'lightgbm'.")

# Define the pipeline with the selected model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Perform randomized search cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_
print("Best Parameters:", best_params)

# Evaluate the best model
accuracy = best_model.score(X_test, y_test)
scores = cross_val_score(best_model, X, y, cv=5)
print("Validation Accuracy:", accuracy)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores)}")

# Train the best model on the entire dataset
best_model.fit(X, y)

# Prepare test data and make predictions
test_data = pd.read_csv('assets/Obesity.csv')
test_data = test_data.drop_duplicates()
test_predictions = best_model.predict(test_data.drop(columns=['NObeyesdad']))

# Note: Assuming test_data has no 'id' column initially; adjust if it does
test_data['id'] = range(len(test_data))
test_predictions = label_encoder.inverse_transform(test_predictions)  # Convert back to original labels

# Create predicted dataframe
predicted_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': test_predictions})

# Save predicted dataframe to CSV
predicted_df.to_csv('predicted.csv', index=False)

# Save the trained model and label encoder
joblib.dump(best_model, 'trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')