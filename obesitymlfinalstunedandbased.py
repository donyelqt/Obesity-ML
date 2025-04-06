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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score
import numpy as np

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
z_scores_train = np.abs(stats.zscore(train_data[numerical_features]))
outliers_train = (z_scores_train > 3).any(axis=1)
train_data = train_data[~outliers_train]

# Encode target variable into numerical labels
label_encoder = LabelEncoder()
train_data['NObeyesdad'] = label_encoder.fit_transform(train_data['NObeyesdad'])

# Add decoded column for visualizations
train_data['NObeyesdad_decoded'] = label_encoder.inverse_transform(train_data['NObeyesdad'])

# Save cleaned train data after preprocessing
train_data.to_csv('cleaned_train_data.csv', index=False)
print("Cleaned train dataset saved as 'cleaned_train_data.csv'")

# Split features and target variable
X = train_data.drop(columns=['NObeyesdad'])
y = train_data['NObeyesdad']

# Train-test split within train_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define data preprocessing steps
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Choose classification type
classification_type = "catboost"

# Select and initialize the model
if classification_type == "rfc":
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif classification_type == "lightgbm":
    model = LGBMClassifier(random_state=42, verbose=-1)
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [-1, 5, 7],
        'classifier__learning_rate': [0.03, 0.05, 0.1],
        'classifier__subsample': [0.9, 1.0],
        'classifier__colsample_bytree': [0.9, 1.0],
        'classifier__min_child_samples': [15, 20, 25],
        'classifier__reg_alpha': [0.0, 0.01],
        'classifier__reg_lambda': [0.0, 0.01]
    }
elif classification_type == "xgboost":
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    param_grid = {
        'classifier__n_estimators': [80, 100, 120, 150],
        'classifier__max_depth': [3, 5, 6, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2, 0.3],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__min_child_weight': [1, 2, 3],
        'classifier__gamma': [0.0, 0.01, 0.05]
    }
elif classification_type == "catboost":
    model = CatBoostClassifier(random_state=42, verbose=0)
    param_grid = {
        'classifier__iterations': [100, 200, 300, 400],
        'classifier__depth': [4, 6, 8],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__l2_leaf_reg': [1, 3, 5],
        'classifier__border_count': [32, 64, 128],
        'classifier__bagging_temperature': [0, 0.5, 1]
    }
elif classification_type == "extratrees":
    model = ExtraTreesClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
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

# Predictions and metrics on X_test
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

6. ROC CURVE:

"""

print(report)

# Save report to file
with open(f'{classification_type}_performance_report.txt', 'w') as f:
    f.write(report)

# Visualizations
plt.figure(figsize=(15, 10))  # Increased figure size to accommodate subplots and labels

# 1. Confusion Matrix Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {classification_type.upper()}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjusted rotation and alignment for better readability
plt.yticks(rotation=0, fontsize=10)

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
    plt.title(f'Top 10 Feature Importances - {classification_type.upper()}')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right', fontsize=10)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig(f'{classification_type}_performance_visualizations.png', bbox_inches='tight')  # Ensure all elements are included in the saved figure
plt.close()  # Close the figure to free memory

# ADDITIONAL: Visualizations
print(f"Generating visualizations for {classification_type.upper()} model...")

# 1. Distribution of Obesity Levels
plt.figure(figsize=(13, 11))
sns.countplot(x='NObeyesdad', data=train_data)
plt.title('Distribution of Obesity Levels')
plt.xlabel('Obesity Level')
plt.ylabel('Count')
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.savefig('target_distribution.png')
plt.close()

# 2. Distribution of Numerical Features
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_data[feature], kde=True)  # Fixed typo: 'figure' to 'feature'
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='NObeyesdad', y=feature, data=train_data)
    plt.title(f'{feature} by Obesity Level')
    plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45)
plt.tight_layout()
plt.savefig('feature_boxplots.png')
plt.close()

plt.figure(figsize=(20, 16))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=feature, hue='NObeyesdad_decoded', data=train_data)
    plt.title(f'{feature} by Obesity Level')
    plt.legend(title='Obesity Level')
plt.tight_layout()
plt.savefig('categorical_feature_target.png')
plt.close()

gender_obesity_counts = train_data.groupby(['Gender', 'NObeyesdad_decoded']).size().unstack()

# Plotting
plt.figure(figsize=(10, 6))
gender_obesity_counts.plot(kind='bar', stacked=True, color=sns.color_palette('Set2'))
plt.title('Obesity Levels by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Individuals')
plt.xticks(rotation=0)
plt.legend(title='Obesity Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('obesity_gender.png')
plt.close()

plt.figure(figsize=(12, 6))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=train_data[feature])
    plt.title(f'Boxplot Outliers of {feature}')
plt.tight_layout()
plt.savefig('boxplotoutliers.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weight', y='Height', hue='NObeyesdad', data=train_data)
plt.title('Weight vs. Height by Obesity Level')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.legend(title='Obesity Level', labels=label_encoder.classes_)
plt.savefig('weight_vs_height_scatter.png')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(train_data[numerical_features].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()

# 9. Actual vs. Predicted Scatter Plot (Corrected)
plt.figure(figsize=(10, 6))
# Prepare data for plotting
plot_data = X_test.copy()

# Transform encoded labels back to strings with error handling
def safe_inverse_transform(encoded_labels, encoder, placeholder="Unknown"):
    encoded_labels = np.array(encoded_labels, dtype=int)
    transformed = np.array([placeholder] * len(encoded_labels), dtype=object)
    valid_mask = (encoded_labels >= 0) & (encoded_labels < len(encoder.classes_))
    if not np.all(valid_mask):
        raise ValueError(f"Invalid labels found: {encoded_labels[~valid_mask]}")
    transformed[valid_mask] = encoder.inverse_transform(encoded_labels[valid_mask])
    return transformed

# Ensure y_test and y_pred are 1D arrays of integers
y_test_flat = y_test.values.ravel().astype(int)
y_pred_flat = y_pred.ravel().astype(int)

plot_data['Actual'] = safe_inverse_transform(y_test_flat, label_encoder)
plot_data['Predicted'] = safe_inverse_transform(y_pred_flat, label_encoder)

# Ensure all values are strings and treat them as categorical
plot_data['Actual'] = pd.Categorical(plot_data['Actual'].astype(str), categories=label_encoder.classes_)
plot_data['Predicted'] = pd.Categorical(plot_data['Predicted'].astype(str), categories=label_encoder.classes_)

plot_data['Correct'] = plot_data['Actual'] == plot_data['Predicted']  # Add correctness column

# Split data into correct and incorrect predictions
correct_data = plot_data[plot_data['Correct']]
incorrect_data = plot_data[~plot_data['Correct']]

# Calculate percentages
total_samples = len(plot_data)
correct_percentage = (len(correct_data) / total_samples * 100) if total_samples > 0 else 0
incorrect_percentage = (len(incorrect_data) / total_samples * 100) if total_samples > 0 else 0

# Plot correct predictions with full opacity
if not correct_data.empty:
    sns.scatterplot(
        data=correct_data,
        x='Weight',
        y='Height',
        hue='Actual',
        style='Predicted',
        alpha=1.0,
        sizes=(40, 40),
        palette='Set2',  # Use a distinct color palette
        markers=['o', '^', 's', 'D', 'v', 'p', '*'],  # Custom shapes
        legend='full'
    )

# Plot incorrect predictions with lower opacity
if not incorrect_data.empty:
    sns.scatterplot(
        data=incorrect_data,
        x='Weight',
        y='Height',
        hue='Actual',
        style='Predicted',
        alpha=0.3,
        sizes=(40, 40),
        palette='Set2',
        markers=['o', '^', 's', 'D', 'v', 'p', '*'],
        legend=False  # Avoid duplicate legend
    )

# Add percentages to the plot
plt.text(
    0.05, 0.95,  # Position in axes coordinates (top-left corner)
    f'Correct: {correct_percentage:.1f}%\nIncorrect: {incorrect_percentage:.1f}%',
    transform=plt.gca().transAxes,  # Use axes coordinates
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
)

plt.title(f'Actual vs Predicted Obesity Levels - {classification_type.upper()} Model')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Obesity Level')
plt.tight_layout()
plt.savefig(f'actual_vs_predicted_scatter_{classification_type.upper()}_Model.png')
plt.close()

# SHAP Summary Plot (for tree-based models)
if classification_type in ['rfc', 'lightgbm', 'xgboost', 'catboost', 'extratrees']:
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
roc_auc_scores = {}  # Dictionary to store AUC scores for each class

for i in range(len(label_encoder.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_scores[label_encoder.classes_[i]] = roc_auc  # Store the AUC score
    plt.plot(fpr, tpr, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')

# Customize the plot
plt.title(f'ROC Curve (One-vs-Rest) - {classification_type.upper()}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Class (AUC)')
plt.tight_layout()
plt.savefig(f'{classification_type}_roc_curve.png')
plt.close()

# Print AUC scores to the console
print("\nROC AUC Scores (One-vs-Rest):")
for class_name, auc_score in roc_auc_scores.items():
    print(f"{class_name}: {auc_score:.2f}")

# Append AUC scores to the performance report file
with open(f'{classification_type}_performance_report_ROC.txt', 'a') as f:
    f.write("\n\nROC AUC Scores (One-vs-Rest):\n")
    for class_name, auc_score in roc_auc_scores.items():
        f.write(f"{class_name}: {auc_score:.2f}\n")

# Precision-Recall Curve with APS in the legend
plt.figure(figsize=(10, 6))
for i in range(len(label_encoder.classes_)):
    # Calculate precision, recall, and APS for the current class
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
    aps = average_precision_score(y_test_bin[:, i], y_prob[:, i])
    # Plot the curve with the APS in the label
    plt.plot(recall, precision, label=f'{label_encoder.classes_[i]} (APS: {aps:.2f})')

# Add a no-skill line (prevalence of the positive class) for reference
for i in range(len(label_encoder.classes_)):
    prevalence = y_test_bin[:, i].mean()
    plt.axhline(y=prevalence, linestyle='--', color='gray', alpha=0.3)

# Customize the plot
plt.title(f'Precision-Recall Curve - {classification_type.upper()}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{classification_type}_precision_recall_curve.png')
plt.close()

# Misclassification Analysis Report
print("Generating Misclassification Analysis Report...")

# Prepare data for misclassification analysis
plot_data = X_test.copy()
y_test_flat = y_test.values.ravel().astype(int)
y_pred_flat = y_pred.ravel().astype(int)

# Transform encoded labels back to strings
plot_data['Actual'] = safe_inverse_transform(y_test_flat, label_encoder)
plot_data['Predicted'] = safe_inverse_transform(y_pred_flat, label_encoder)

# Ensure all values are strings and treat them as categorical
plot_data['Actual'] = pd.Categorical(plot_data['Actual'].astype(str), categories=label_encoder.classes_)
plot_data['Predicted'] = pd.Categorical(plot_data['Predicted'].astype(str), categories=label_encoder.classes_)

plot_data['Correct'] = plot_data['Actual'] == plot_data['Predicted']
incorrect_data = plot_data[~plot_data['Correct']]

# 1. Heatmap of Misclassified Samples (Actual vs. Predicted NObeyesdad Classes)
plt.figure(figsize=(10, 8))
if not incorrect_data.empty:
    # Create a cross-tabulation of actual vs. predicted classes for misclassified samples
    misclassification_matrix = pd.crosstab(
        incorrect_data['Actual'], 
        incorrect_data['Predicted'], 
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    # Ensure all classes are included, even if they have no misclassifications
    for class_name in label_encoder.classes_:
        if class_name not in misclassification_matrix.index:
            misclassification_matrix.loc[class_name] = 0
        if class_name not in misclassification_matrix.columns:
            misclassification_matrix[class_name] = 0
    # Reorder the matrix to match the label_encoder.classes_ order
    misclassification_matrix = misclassification_matrix.reindex(
        index=label_encoder.classes_, 
        columns=label_encoder.classes_, 
        fill_value=0
    )
    # Plot the heatmap
    sns.heatmap(
        misclassification_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Reds', 
        cbar_kws={'label': 'Number of Misclassifications'},
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Misclassification Heatmap (Actual vs. Predicted) - {classification_type.upper()}')
    plt.xlabel('Predicted NObeyesdad')
    plt.ylabel('Actual NObeyesdad')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{classification_type}_misclassification_heatmap.png')
    plt.close()
else:
    print("No misclassified samples to plot.")

# 2. Error Rate per Class
error_rates = incorrect_data.groupby('Actual').size() / plot_data.groupby('Actual').size()
error_rates = error_rates.fillna(0)  # Replace NaN with 0 for classes with no errors
plt.figure(figsize=(10, 6))
error_rates.plot(kind='bar', color='salmon')
plt.title(f'Error Rate per Class - {classification_type.upper()}')
plt.xlabel('Actual Class')
plt.ylabel('Error Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{classification_type}_error_rate_per_class.png')
plt.close()

# 3. Text Summary of Misclassification Analysis
misclassification_summary = f"\n8. Misclassification Analysis:\n"
misclassification_summary += f"- Total Misclassified Samples: {len(incorrect_data)}\n"
misclassification_summary += f"- Overall Misclassification Rate: {len(incorrect_data) / len(plot_data):.4f}\n"
misclassification_summary += f"- Refer to the Confusion Matrix (in '{classification_type}_performance_visualizations.png') for a detailed breakdown of true positives, false positives, false negatives, and true negatives.\n"
misclassification_summary += f"- Refer to the Misclassification Heatmap ('{classification_type}_misclassification_heatmap.png') for a focused view of actual vs. predicted NObeyesdad classes for misclassified samples.\n"
misclassification_summary += f"- Refer to the Actual vs. Predicted Scatter Plot ('actual_vs_predicted_scatter_{classification_type.upper()}_Model.png') to see misclassified samples in the Weight vs. Height feature space.\n"

# Misclassifications per class
misclassification_summary += "\nMisclassifications per Class:\n"
for class_name in label_encoder.classes_:
    misclassified_count = len(incorrect_data[incorrect_data['Actual'] == class_name])
    total_count = len(plot_data[plot_data['Actual'] == class_name])
    misclassification_rate = misclassified_count / total_count if total_count > 0 else 0
    misclassification_summary += f"- {class_name}: {misclassified_count} misclassified out of {total_count} (Rate: {misclassification_rate:.4f})\n"

# Common misclassifications from the confusion matrix
common_misclassifications = incorrect_data.groupby(['Actual', 'Predicted']).size().sort_values(ascending=False)
if not common_misclassifications.empty:
    misclassification_summary += "\nCommon Misclassifications:\n"
    for (actual, predicted), count in common_misclassifications.items():
        misclassification_summary += f"- {actual} misclassified as {predicted}: {count} times\n"
else:
    misclassification_summary += "\nNo misclassifications to analyze.\n"

# Potential reasons for misclassifications
misclassification_summary += "\nPotential Reasons for Misclassifications:\n"
misclassification_summary += "- Class Imbalance: Classes with fewer samples may have higher error rates due to underrepresentation.\n"
misclassification_summary += "- Feature Overlap: Classes with similar feature distributions (e.g., Overweight_Level_I and Overweight_Level_II) may be harder to distinguish.\n"
misclassification_summary += "- Model Limitations: The model may struggle with boundary cases where features like Weight and Height are close to decision thresholds.\n"

# Append to the performance report
with open(f'{classification_type}_performance_report.txt', 'a') as f:
    f.write(misclassification_summary)

print("Misclassification Analysis Report added to the performance report.")

print(f"Visualizations saved with prefix '{classification_type}_' (e.g., '{classification_type}_performance_visualizations.png')")

# Train the best model on the entire dataset
best_model.fit(X, y)

# Use X_test for predictions 
test_predictions = best_model.predict(X_test)

# Create a DataFrame with IDs and predictions
test_data_with_predictions = X_test.copy()
test_data_with_predictions['id'] = range(len(test_data_with_predictions))
test_data_with_predictions['NObeyesdad'] = label_encoder.inverse_transform(test_predictions)

# Save predicted dataframe
predicted_df = test_data_with_predictions[['id', 'NObeyesdad']]
predicted_df.to_csv('predicted.csv', index=False)

# Save the trained model and label encoder
joblib.dump(best_model, f'{classification_type}_trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print(f"Report saved as '{classification_type}_performance_report.txt'")
print(f"Visualizations saved as '{classification_type}_performance_visualizations.png'")