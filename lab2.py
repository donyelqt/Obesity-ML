import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the data
train_data = pd.read_csv('Obesity.csv')
train_data = train_data.drop_duplicates()

# Remove duplicated
train_data = train_data.drop_duplicates()

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Encode target variable
label_encoder = LabelEncoder()
train_data['NObeyesdad'] = label_encoder.fit_transform(train_data['NObeyesdad'])

# Split features and target variable
X = train_data.drop(columns=['NObeyesdad'])
y = train_data['NObeyesdad']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])

# Define hyperparameters for randomized search
param_grid = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Perform randomized search cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Evaluate the best model on the validation set
accuracy = best_model.score(X_test, y_test)
print("Validation Accuracy:", accuracy)

# Train the best model on the entire dataset
best_model.fit(X, y)

# Prepare test data
test_data = pd.read_csv('Obesity.csv')
test_data = test_data.drop_duplicates()

# Make predictions
test_predictions = best_model.predict(test_data.drop(columns=['NObeyesdad'], errors='ignore'))
test_data['id'] = range(len(test_data))
predicted_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': label_encoder.inverse_transform(test_predictions)})

# Save predicted dataframe to CSV
predicted_df.to_csv('predicted.csv', index=False)

# Save the trained model
joblib.dump(best_model, 'trained_rf_model')
joblib.dump(label_encoder, 'label_encoder')
