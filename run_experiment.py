# This script is used to run the experiments on metritis risk classification

import argparse
import os
import getpass
import shutil
import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.utils._tags import _safe_tags  # Import for patching
import shutil

# Input arguments
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--two_classes', type=bool, default=True,
                    help='If True, the dataset is reduced to two classes (healthy and metritis)')
parser.add_argument('--train_farm', type=str, default='A',help='Farm to train on')
parser.add_argument('--use_same_data', type=bool, default=False,
                    help='If True, the same data is used for training and testing')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
data = pd.read_csv(os.path.join('data','dataset_20240614.csv'))

# Preprocess data
data = data.dropna()  # Dropping rows with missing values for simplicity

# Feature engineering
#    - **Dynamic Variables:** Features that vary over time (e.g., `DIM`, `THI`, `EstFeeding`) are treated as time-series data.
#    - **Static Variables:** Features that remain constant for a given cow (e.g., `LactNo`, `ColCalvInt`, `RP`, `VLS`).
#    - Time-series data for each cow is processed:
#      - Dynamic variables are normalized using `StandardScaler`.
#      - Sequences are padded to ensure uniform length across all cows.
#    - **Statistical Representation:**
#      - Summary statistics (mean, standard deviation, min, max, kurtosis, and skewness) are computed for the dynamic variables to capture their distributional properties.
#    - Combined Feature Matrix:
#      - The statistical representations of the dynamic variables are concatenated with static variables to form the final feature matrix.
#    - Labels (`NewClassifier`) are encoded into numerical values for compatibility with machine learning models (`Healthy: 0`, `MildMetritis: 1`, `AcuteMetritis: 2`).

# Define features and target
static_vars = ['LactNo', 'ColCalvInt', 'Male', 'StillBirth', 'Twins', 'Dystocia', 'RP', 'VLS']
dynamic_vars = ['DIM', 'BCSCollar', '∆BCS', 'THI', 'EstFeeding', 'EstRum', 'EstIdle', 'EstActive']
target_var = 'NewClassifier'

# Extract unique cow IDs
cow_ids = data['ID'].unique()

# Create dictionaries to hold static and dynamic data
static_data = {}
dynamic_data = {}

# Iterate over each cow ID to separate static and dynamic data
for cow_id in cow_ids:
    cow_data = data[data['ID'] == cow_id]
    static_data[cow_id] = cow_data[static_vars].iloc[0].values
    dynamic_data[cow_id] = cow_data[dynamic_vars].values

# Normalize dynamic variables
scaler = StandardScaler()
for cow_id in cow_ids:
    dynamic_data[cow_id][:, 1:] = scaler.fit_transform(dynamic_data[cow_id][:, 1:])  # Normalize except first dynamic var

# Determine the maximum sequence length
max_len = max([len(dynamic_data[cow_id]) for cow_id in cow_ids])

# Pad sequences
for cow_id in cow_ids:
    dynamic_data[cow_id] = np.pad(dynamic_data[cow_id], ((0, max_len - len(dynamic_data[cow_id])), (0, 0)), 'constant')

# Compute statistics for dynamic data
def compute_statistics(dynamic_data):
    statistics = {}
    for cow_id, data in dynamic_data.items():
        statistics[cow_id] = np.hstack([
            np.mean(data, axis=0),
            np.std(data, axis=0),
            np.min(data, axis=0),
            np.max(data, axis=0),
            pd.DataFrame(data).kurtosis(axis=0).values,
            pd.DataFrame(data).skew(axis=0).values
        ])
    return statistics

dynamic_statistics = compute_statistics(dynamic_data)

# Combine features (excluding FarmC)
combined_features = np.array([np.concatenate((dynamic_statistics[cow_id], static_data[cow_id])) for cow_id in cow_ids if cow_id in dynamic_statistics])
labels = np.array([data[data['ID'] == cow_id][target_var].values[0] for cow_id in cow_ids if cow_id in dynamic_statistics])

# Manually encode labels to ensure correct mapping
label_mapping = {'Healthy': 0, 'MildMetritis': 1, 'AcuteMetritis': 2}
labels = np.array([label_mapping[label] for label in labels])

if args.two_classes:
    label_mapping = {'Healthy': 0, 'Metritis': 1}
    labels = np.where(labels > 0, 1, 0)

#### **Model Building**
#    - A variety of machine learning classifiers are initialized, including:
#      - Random Forest, Logistic Regression, Decision Trees, SVM, KNN, XGBoost, Gradient Boosting, AdaBoost, and Extra Trees.
#    - These classifiers are used to explore different algorithms' performance on the classification task.

# Define the patch to add __sklearn_tags__
def __sklearn_tags__(self):
    tags = _safe_tags(self, key="estimator_type")
    tags.append("classifier")  # Add "classifier" tag
    return tags

# Apply the patch to XGBClassifier
XGBClassifier.__sklearn_tags__ = __sklearn_tags__

# Define a function to directly fit XGBClassifier
def fit_xgb_classifier(clf, X, y):
    """Fits an XGBClassifier without RandomizedSearchCV."""
    if isinstance(clf, XGBClassifier):
        clf.fit(X, y)
    return clf

# Apply the patch to XGBClassifier
XGBClassifier.__sklearn_tags__ = __sklearn_tags__

# Define a function to directly fit XGBClassifier
def fit_xgb_classifier(clf, X, y):
    """Fits an XGBClassifier without RandomizedSearchCV."""
    if isinstance(clf, XGBClassifier):
        clf.fit(X, y)
    return clf

# Initialize base classifiers with the wrapper
base_classifiers = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=2000, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('xgb', XGBClassifier(random_state=42)),  # Use regular XGBClassifier
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42)),
    ('et', ExtraTreesClassifier(random_state=42))
]

import matplotlib.pyplot as plt

# Create the histogram
plt.figure(figsize=(8, 6))
plt.hist(labels, bins=range(len(label_mapping) + 1), align='left', rwidth=0.8)  # Use label_mapping for bins
plt.xticks(range(len(label_mapping)), label_mapping.keys())  # Set x-axis ticks and labels
plt.xlabel("Metritis Risk Level")
plt.ylabel("Number of Cows")
plt.title("Distribution of Metritis Risk Levels")
plt.show()

# Model training and hyperparameter tuning (via `RandomizedSearchCV`)

from scipy.stats import randint, uniform  # Import randint and uniform

# Select which farm to use for training
train_farm = 'A'  # Change to 'B' to use Farm B for training

# Option to use the same data for testing
use_same_data = False # Set to True if you want to use the same data for both training and testing

farm_mapping = data.drop_duplicates(subset='ID').set_index('ID')['FarmC'].to_dict()

if use_same_data:
    if train_farm == 'A':
        train_mask = np.array([farm_mapping[cow_id] == 'A' for cow_id in cow_ids if cow_id in dynamic_statistics])
        test_mask = train_mask
    else:
        train_mask = np.array([farm_mapping[cow_id] == 'B' for cow_id in cow_ids if cow_id in dynamic_statistics])
        test_mask = train_mask
    train_test_label = f"train_{train_farm}_test_same"
else:
    if train_farm == 'A':
        train_mask = np.array([farm_mapping[cow_id] == 'A' for cow_id in cow_ids if cow_id in dynamic_statistics])
        test_mask = np.array([farm_mapping[cow_id] == 'B' for cow_id in cow_ids if cow_id in dynamic_statistics])
        train_test_label = "train_A_test_B"
    else:
        train_mask = np.array([farm_mapping[cow_id] == 'B' for cow_id in cow_ids if cow_id in dynamic_statistics])
        test_mask = np.array([farm_mapping[cow_id] == 'A' for cow_id in cow_ids if cow_id in dynamic_statistics])
        train_test_label = "train_B_test_A"

X_train = combined_features[train_mask]
y_train = labels[train_mask]

X_test = combined_features[test_mask]
y_test = labels[test_mask]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter distributions for the classifiers (TabPFN does not need hyperparameter tuning)
param_distributions = {
    'rf': {
        'n_estimators': randint(100, 200),
        'max_depth': [None, 10, 20],
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 2),
    },
    'svm': {
        'C': [1e-2, 1e-1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear'],
    },
    'knn': {
        'n_neighbors': randint(3, 7),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'xgb': {
        'n_estimators': randint(100, 200),
        'learning_rate': uniform(0.01, 0.19),
        'max_depth': randint(3, 7)
    },
    'gb': {
        'n_estimators': randint(100, 200),
        'learning_rate': uniform(0.01, 0.19),
        'max_depth': randint(3, 7)
    },
    'ada': {
        'n_estimators': randint(50, 1000),
        'learning_rate': uniform(0.01, 0.99)
    },
    'et': {
        'n_estimators': randint(100, 200),
        'max_depth': [None, 10, 20],
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 2)
    }
}

# Perform hyperparameter tuning with RandomizedSearchCV
print("Starting hyperparameter tuning...")
best_classifiers = {}
for name, clf in base_classifiers:
    print(f"Tuning hyperparameters for {name}...")
    if name in param_distributions and name != 'xgb':  # Exclude 'xgb'
        random_search = RandomizedSearchCV(clf, param_distributions[name], n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)
        best_classifiers[name] = random_search.best_estimator_
        print(f"Best parameters for {name}: {random_search.best_params_}")
    else:
        best_classifiers[name] = fit_xgb_classifier(clf, X_train, y_train) if name == 'xgb' else clf.fit(X_train, y_train) # Fit XGBClassifier separately or others
    print(f"Finished tuning {name}")

    # Check if the 'results' directory exists, and create it if it doesn't
if not os.path.exists('results'):
    os.makedirs('results')  # Use os.makedirs to create nested directories if needed

# Collect predictions from individual classifiers
print("Collecting predictions...")
predictions = {'ID': cow_ids[test_mask], 'Actual': y_test}
for name, clf in best_classifiers.items():
    predictions[name] = clf.predict(X_test)
print("Finished collecting predictions")

# Create DataFrame with the collected predictions
print("Creating DataFrame...")
predictions_df = pd.DataFrame(predictions)

# Save the DataFrames to CSV files
print(f"Saving DataFrames to CSV files with label {train_test_label}...")
predictions_df.to_csv(os.path.join('results',
                      f'class_predictions_{train_test_label}.csv'), index=False)

print(predictions_df.head())

# Compute performance measures with bootstrapping
print("Computing performance measures with bootstrapping...")
performance_measures = []
n_bootstraps = 1000
rng = np.random.RandomState(42)

for name in predictions_df.columns[2:]:  # Skip 'ID' and 'Actual' columns
    bootstrapped_accuracies = []
    bootstrapped_precisions = []
    bootstrapped_recalls = []
    bootstrapped_f1s = []
    bootstrapped_aucs = []
    bootstrapped_sensitivities = []
    bootstrapped_specificities = []
    bootstrapped_ppvs = []
    bootstrapped_npvs = []
    pred_labels = predictions_df[name]

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(pred_labels), len(pred_labels))
        if len(np.unique(predictions_df['Actual'].iloc[indices])) < 2:
            # Skip this sample because it has less than two classes
            continue
        # Convert relevant columns to dense numpy arrays
        actual_values = predictions_df['Actual'].iloc[indices].to_numpy(dtype=int)  # Ensure integer type
        predicted_values = pred_labels.iloc[indices]

        # Compute metrics
        score_acc = accuracy_score(actual_values, predicted_values)
        score_prec = precision_score(actual_values, predicted_values, average='weighted', zero_division=1)
        score_rec = recall_score(actual_values, predicted_values, average='weighted', zero_division=1)
        score_f1 = f1_score(actual_values, predicted_values, average='weighted', zero_division=1)

        # Compute AUC only if probability estimates are available
        if hasattr(best_classifiers[name], 'predict_proba'):
            # Adjust indexing based on the type of X_test
            if isinstance(X_test, pd.DataFrame):
                proba_values = best_classifiers[name].predict_proba(X_test.iloc[indices])
            else:  # Assume X_test is a NumPy array
                proba_values = best_classifiers[name].predict_proba(X_test[indices])

            # For binary classification, use the probability of the positive class
            if proba_values.shape[1] == 2:  # Binary classification
                score_auc = roc_auc_score(actual_values, proba_values[:, 1])
            else:  # Multiclass classification
                score_auc = roc_auc_score(actual_values, proba_values, multi_class='ovr')
        else:
            score_auc = np.nan  # Skip AUC calculation if predict_proba is not available

        # Compute the confusion matrix for multiclass classification
        cm = confusion_matrix(actual_values, predicted_values)

        # Compute metrics for each class
        sensitivity_list = []
        specificity_list = []
        ppv_list = []
        npv_list = []

        for i in range(cm.shape[0]):  # Iterate over each class
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            ppv_list.append(ppv)
            npv_list.append(npv)

        # Average the metrics across all classes (macro-average)
        score_sensitivity = np.mean(sensitivity_list)
        score_specificity = np.mean(specificity_list)
        score_ppv = np.mean(ppv_list)
        score_npv = np.mean(npv_list)

        # Append metrics to bootstrapped lists
        bootstrapped_accuracies.append(score_acc)
        bootstrapped_precisions.append(score_prec)
        bootstrapped_recalls.append(score_rec)
        bootstrapped_f1s.append(score_f1)
        bootstrapped_aucs.append(score_auc)
        bootstrapped_sensitivities.append(score_sensitivity)
        bootstrapped_specificities.append(score_specificity)
        bootstrapped_ppvs.append(score_ppv)
        bootstrapped_npvs.append(score_npv)

    # Compute 95% confidence intervals
    def calculate_ci(values):
        return (np.percentile(values, 2.5), np.percentile(values, 97.5))

    performance_measures.append({
        'Classifier': name,
        'Accuracy': np.mean(bootstrapped_accuracies),
        'Accuracy CI': calculate_ci(bootstrapped_accuracies),
        'Precision': np.mean(bootstrapped_precisions),
        'Precision CI': calculate_ci(bootstrapped_precisions),
        'Recall': np.mean(bootstrapped_recalls),
        'Recall CI': calculate_ci(bootstrapped_recalls),
        'F1-Score': np.mean(bootstrapped_f1s),
        'F1-Score CI': calculate_ci(bootstrapped_f1s),
        'AUC': np.mean(bootstrapped_aucs),
        'AUC CI': calculate_ci(bootstrapped_aucs),
        'Sensitivity': np.mean(bootstrapped_sensitivities),
        'Sensitivity CI': calculate_ci(bootstrapped_sensitivities),
        'Specificity': np.mean(bootstrapped_specificities),
        'Specificity CI': calculate_ci(bootstrapped_specificities),
        'PPV': np.mean(bootstrapped_ppvs),
        'PPV CI': calculate_ci(bootstrapped_ppvs),
        'NPV': np.mean(bootstrapped_npvs),
        'NPV CI': calculate_ci(bootstrapped_npvs)
    })

# Create a DataFrame for the performance measures
performance_df = pd.DataFrame(performance_measures)

# Save the performance measures DataFrame to a CSV file
print(f"Saving performance measures DataFrame to CSV file with label {train_test_label}...")
performance_df.to_csv(os.path.join('results',
                      f'class_performance_measures_{train_test_label}.csv'), index=False)

print(performance_df)
print("Finished computing and saving performance measures.")

import pickle

# ... after you have defined/created these variables ...
variables_to_save = {
    'best_classifiers': best_classifiers,
    'X_test': X_test,
    'y_test': y_test,
    'dynamic_vars': dynamic_vars,
    'static_vars': static_vars,
    'train_test_label': train_test_label
}

save_path = 'variables.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(variables_to_save, f)