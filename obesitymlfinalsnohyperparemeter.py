import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
train_data = pd.read_csv('assets/Obesity.csv')

# Display first few rows
print("Dataset Preview:")
print(train_data.head())

# Check for missing values
print("\nMissing Values:")
print(train_data.isnull().sum())

# Remove duplicates
train_data = train_data.drop_duplicates()

# Define categorical and numerical features
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Handle outliers using Z-score
z_scores = np.abs(stats.zscore(train_data[numerical_features]))
outliers = (z_scores > 3).any(axis=1)
train_data = train_data[~outliers]

# Encode target variable into numerical labels
label_encoder = LabelEncoder()
train_data['NObeyesdad'] = label_encoder.fit_transform(train_data['NObeyesdad'])

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
classification_type = "catboost"  # Options: 'rfc', 'svc', 'kneighbors', 'logistic', 'lightgbm', 'xgboost', 'catboost', 'extratrees'

# Select and initialize the model with default parameters
if classification_type == "rfc":
    model = RandomForestClassifier(random_state=42)
elif classification_type == "svc":
    model = SVC(random_state=42)
elif classification_type == "kneighbors":
    model = KNeighborsClassifier()
elif classification_type == "logistic":
    model = LogisticRegression(random_state=42, max_iter=1000)
elif classification_type == "lightgbm":
    model = LGBMClassifier(random_state=42, verbose=-1)
elif classification_type == "xgboost":
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
elif classification_type == "catboost":
    model = CatBoostClassifier(random_state=42, verbose=0)
elif classification_type == "extratrees":
    model = ExtraTreesClassifier(random_state=42)
else:
    raise ValueError("Invalid classification type. Choose from 'rfc', 'svc', 'kneighbors', 'logistic', 'lightgbm', 'xgboost', 'catboost', 'extratrees'.")

# Define the pipeline with the selected model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Fit the model without hyperparameter tuning
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(pipeline, X, y, cv=5)

# Generate detailed report
report = f"""
Model Performance Report (Default Parameters)
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {classification_type.upper()}

1. Evaluation Metrics:
- Accuracy: {accuracy:.4f}
- Mean Squared Error: {mse:.4f}
- R² Score: {r2:.4f}
- Cross-validation Scores: {cv_scores}
- Mean CV Score: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})

2. Classification Report:
{class_report}

3. Interpretation:
- Accuracy indicates the proportion of correct predictions
- MSE measures the average squared difference between predicted and actual values
- R² shows the proportion of variance explained by the model
- Cross-validation provides a robust estimate of model performance
"""

print(report)

# Save report to file
with open(f'{classification_type}_default_performance_report.txt', 'w') as f:
    f.write(report)

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Confusion Matrix Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. Cross-validation scores
plt.subplot(2, 2, 2)
plt.bar(range(1, 6), cv_scores)
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# 3. Feature Importance (if applicable)
if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
    feature_names = (numerical_features + 
                    list(pipeline.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(categorical_features)))
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    plt.subplot(2, 2, 3)
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]
    feature_imp.plot(kind='bar')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{classification_type}_default_performance_visualizations.png')
plt.close()

# Train the model on the entire dataset
pipeline.fit(X, y)

# Prepare test data and make predictions
test_data = pd.read_csv('assets/Obesity.csv')
test_data = test_data.drop_duplicates()
test_predictions = pipeline.predict(test_data.drop(columns=['NObeyesdad']))

# Add IDs and decode predictions
test_data['id'] = range(len(test_data))
test_predictions = label_encoder.inverse_transform(test_predictions)

# Create and save predicted dataframe
predicted_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': test_predictions})
predicted_df.to_csv('predicted_default.csv', index=False)

# Save the trained model and label encoder
joblib.dump(pipeline, 'trained_model_default.pkl')
joblib.dump(label_encoder, 'label_encoder_default.pkl')

print(f"Report saved as '{classification_type}_default_performance_report.txt'")
print(f"Visualizations saved as '{classification_type}_default_performance_visualizations.png'")