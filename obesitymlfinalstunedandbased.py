import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

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

# Handle outliers and remove using Z-score
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
classification_type = "lightgbm"  # Options: 'rfc', 'svc', 'kneighbors', 'logistic', 'lightgbm', 'xgboost', 'catboost', 'extratrees'

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
        'classifier__solver': ['liblinear']
    }
elif classification_type == "lightgbm":
    model = LGBMClassifier(random_state=42, verbose=-1)
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],  # More trees for capacity
        'classifier__max_depth': [-1, 5, 7],  # Balanced depth options
        'classifier__learning_rate': [0.03, 0.05, 0.1],  # Lower rates for better optimization
        'classifier__subsample': [0.9, 1.0],  # Close to default
        'classifier__colsample_bytree': [0.9, 1.0],  # Close to default
        'classifier__min_child_samples': [15, 20, 25],  # Around default
        'classifier__reg_alpha': [0.0, 0.01],  # Light regularization
        'classifier__reg_lambda': [0.0, 0.01]  # Light regularization
    }
elif classification_type == "xgboost":
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    param_grid = {
        'classifier__n_estimators': [80, 100, 120, 150],  # Center around default (100)
        'classifier__max_depth': [3, 5, 6, 7],  # Include default (6) and nearby values
        'classifier__learning_rate': [0.05, 0.1, 0.2, 0.3],  # Include default (0.3) and lower values
        'classifier__subsample': [0.8, 0.9, 1.0],  # Default is 1.0, test slight subsampling
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],  # Default is 1.0, test slight subsampling
        'classifier__min_child_weight': [1, 2, 3],  # Default is 1, light regularization
        'classifier__gamma': [0.0, 0.01, 0.05]  # Default is 0, minimal regularization
    }
elif classification_type == "catboost":
    model = CatBoostClassifier(random_state=42, verbose=0)
    param_grid = {
        'classifier__iterations': [100, 200, 300, 400],  # More reasonable range
        'classifier__depth': [4, 6, 8],  # Avoid too deep trees
        'classifier__learning_rate': [0.01, 0.05, 0.1],  # Focus on smaller rates
        'classifier__l2_leaf_reg': [1, 3, 5],  # Moderate regularization
        'classifier__border_count': [32, 64, 128],  # Standard range
        'classifier__bagging_temperature': [0, 0.5, 1]  # Control overfitting
    }
elif classification_type == "extratrees":
    model = ExtraTreesClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]  # Fixed: Removed 'auto', added None
    }
else:
    raise ValueError("Invalid classification type. Choose from 'rfc', 'svc', 'kneighbors', 'logistic', 'lightgbm', 'xgboost', 'catboost', 'extratrees'.")

# Define the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Baseline performance (default parameters)
baseline_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
baseline_pipeline.fit(X_train, y_train)
baseline_accuracy = baseline_pipeline.score(X_test, y_test)
baseline_cv_scores = cross_val_score(baseline_pipeline, X, y, cv=5)

# Perform randomized search cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Predictions and metrics
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(best_model, X, y, cv=5)

# Confusion matrix analysis
cm_analysis = ""
for i, label in enumerate(label_encoder.classes_):
    false_positives = conf_matrix[:, i].sum() - conf_matrix[i, i]
    false_negatives = conf_matrix[i, :].sum() - conf_matrix[i, i]
    if false_positives > 0 or false_negatives > 0:
        cm_analysis += f"- {label}: {false_positives} false positives, {false_negatives} false negatives, indicating potential overlap with similar classes.\n"

# Feature importance analysis (for tree-based models)
feature_analysis = ""
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    feature_names = (numerical_features + 
                    list(best_model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .named_steps['onehot']
                        .get_feature_names_out(categorical_features)))
    importances = best_model.named_steps['classifier'].feature_importances_
    top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:5]
    feature_analysis = "Top 5 Important Features:\n" + "\n".join(
        f"- {feat}: {imp:.4f} (likely critical for obesity classification)" 
        for feat, imp in top_features.items()
    )
    
# Generate detailed report
report = f"""
Model Performance Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {classification_type.upper()}

1. Hyperparameter Tuning Results:
- Baseline Accuracy (Default Parameters): {baseline_accuracy:.4f}
- Tuned Accuracy: {accuracy:.4f}
- Improvement: {(accuracy - baseline_accuracy):.4f}
- Baseline CV Mean Score: {np.mean(baseline_cv_scores):.4f} (± {np.std(baseline_cv_scores):.4f})
- Tuned CV Mean Score: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})
- Best Parameters: {', '.join([f'{k}: {v}' for k, v in best_params.items()])}
- Tuning Process: RandomizedSearchCV with 50 iterations and 3-fold CV was used to explore a wide parameter space efficiently. 
  The optimal settings balance model complexity and generalization, with parameters like learning rate and depth tuned to prevent overfitting.

2. Evaluation Metrics:
- Accuracy: {accuracy:.4f}
- Mean Squared Error: {mse:.4f}
- R² Score: {r2:.4f}
- Cross-validation Scores: {cv_scores}
- Mean CV Score: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})

3. Classification Report:
{class_report}

4. Model Behavior Analysis:
- Confusion Matrix Insights:
{cm_analysis if cm_analysis else "- No significant misclassifications observed."}
- {feature_analysis if feature_analysis else "Feature importance not available for this model."}
- General Behavior: {classification_type.upper()} {'handles categorical features natively, potentially improving performance over one-hot encoded models' if classification_type == 'catboost' else 'relies on feature preprocessing and may be sensitive to parameter settings.'}

5. Interpretations:
- Accuracy improvement suggests tuning enhanced generalization.
- Low CV score variance ({np.std(cv_scores):.4f}) indicates robust performance across data splits.
- MSE and R² reflect prediction consistency, though more relevant for regression tasks.
- Confusion matrix patterns and feature importance (if applicable) highlight key decision factors in obesity classification.
"""

print(report)

# Save report to file
with open(f'{classification_type}_performance_report.txt', 'w') as f:
    f.write(report)

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Confusion Matrix Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {classification_type.upper()}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. Cross-validation scores comparison
plt.subplot(2, 2, 2)
plt.plot(range(1, 6), baseline_cv_scores, label='Baseline', marker='o')
plt.plot(range(1, 6), cv_scores, label='Tuned', marker='o')
plt.title(f'Cross-Validation Scores: Baseline vs Tuned - {classification_type.upper()}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# 3. Feature Importance
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    plt.subplot(2, 2, 3)
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]
    feature_imp.plot(kind='bar')
    plt.title(f'Top 10 Feature Importances -{classification_type.upper()}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{classification_type}_performance_visualizations.png')
plt.close()

# ADDITIONAL: Visualizations
print(f"Generating visualizations for {classification_type.upper()} model...")

# 1. EDA Visuals (Before Preprocessing, so place this earlier if desired)
# Histogram of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='NObeyesdad', data=train_data)
plt.title('Distribution of Obesity Levels')
plt.xlabel('Obesity Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('target_distribution.png')
plt.close()

# Histograms for numerical feature distribution
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Add Box Plots of target variable
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='NObeyesdad', y=feature, data=train_data)
    plt.title(f'{feature} by Obesity Level')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_boxplots.png')
plt.close()

# Boxplot of Outliers
plt.figure(figsize=(12, 6))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=train_data[feature])
    plt.title(f'Boxplot Outliers of {feature}')
plt.tight_layout()
plt.show()

# Scatter plots for selected numerical feature pairs vs. target
# Example: Weight vs. Height colored by obesity level
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight', y='Height', hue='NObeyesdad', data=train_data)
plt.title('Weight vs. Height by Obesity Level')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.savefig('weight_vs_height_scatter.png')
plt.close()

# Correlation Heatmap of Numerical Features Relationships
plt.figure(figsize=(10, 8))
sns.heatmap(train_data[numerical_features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()

# 2. Model-Specific Visualizations
plt.figure(figsize=(15, 10))

# Confusion Matrix Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {classification_type.upper()}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Cross-validation Scores Comparison
plt.subplot(2, 2, 2)
plt.plot(range(1, 6), baseline_cv_scores, label='Baseline', marker='o')
plt.plot(range(1, 6), cv_scores, label='Tuned', marker='o')
plt.title(f'Cross-Validation Scores: Baseline vs Tuned - {classification_type.upper()}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# Feature Importance (if applicable)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    plt.subplot(2, 2, 3)
    feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]
    feature_imp.plot(kind='bar')
    plt.title(f'Top 10 Feature Importances - {classification_type.upper()}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{classification_type}_performance_visualizations.png')
plt.close()

# 3. Best Model Insights
# SHAP Summary Plot (for tree-based models)
if classification_type in ['rfc', 'lightgbm', 'xgboost', 'catboost', 'extratrees', 'gradientboost']:
    explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
    transformed_X = best_model.named_steps['preprocessor'].transform(X_test)
    shap_values = explainer.shap_values(transformed_X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, transformed_X, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {classification_type.upper()}')
    plt.savefig(f'{classification_type}_shap_summary.png')
    plt.close()

# ROC Curve (Multi-Class)
y_test_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
y_prob = best_model.predict_proba(X_test)
plt.figure(figsize=(10, 6))
for i in range(len(label_encoder.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f'ROC Curve (One-vs-Rest) - {classification_type.upper()}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(f'{classification_type}_roc_curve.png')
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(10, 6))
for i in range(len(label_encoder.classes_)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(recall, precision, label=f'{label_encoder.classes_[i]}')
plt.title(f'Precision-Recall Curve - {classification_type.upper()}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(f'{classification_type}_precision_recall_curve.png')
plt.close()

print(f"Visualizations saved with prefix '{classification_type}_' (e.g., '{classification_type}_performance_visualizations.png')")

# 1.) Datasets Trends of target variable
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='NObeyesdad', y=feature, data=train_data)
    plt.title(f'{feature} by Obesity Level')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_target_boxplots.png')
plt.close()

# 2.)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=feature, hue='NObeyesdad', data=train_data)
    plt.title(f'{feature} by Obesity Level')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_feature_target.png')
plt.close()

# Train the best model on the entire dataset
best_model.fit(X, y)

# Prepare test data and make predictions
test_data = pd.read_csv('assets/Obesity.csv')
test_data = test_data.drop_duplicates()
test_predictions = best_model.predict(test_data.drop(columns=['NObeyesdad']))

# Add IDs and decode predictions
test_data['id'] = range(len(test_data))
test_predictions = label_encoder.inverse_transform(test_predictions)

# Create and save predicted dataframe
predicted_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': test_predictions})
predicted_df.to_csv('predicted.csv', index=False)

# Save the trained model and label encoder with model-specific naming
joblib.dump(best_model, f'{classification_type}_trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"Report saved as '{classification_type}_performance_report.txt'")
print(f"Visualizations saved as '{classification_type}_performance_visualizations.png'")    