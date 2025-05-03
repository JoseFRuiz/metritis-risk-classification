# This section provides a detailed exploration of our metritis risk classification models' decision-making process through a comprehensive explainability analysis. We employ several techniques to understand how different features influence model predictions.

# First, we utilize **permutation importance** to quantify each feature's contribution. Permutation importance measures the decrease in model performance (specifically, the F1-score in our case) when a feature's values are randomly shuffled. This provides a global understanding of feature importance. Mathematically, for a feature *j*, we calculate the permutation importance as:
# content_copy
# download
# Use code with caution.
# Markdown

# Importance(j) = Mean( score(original model) - score(model with feature j permuted) )

# Where `score` is the F1-score and we average the score differences over several random permutations.

# Second, we employ **SHAP (SHapley Additive exPlanations) values**, a model-agnostic method that provides both global and local explanations. SHAP values quantify each feature's contribution to a specific prediction relative to the average prediction. In our specific case we are using the `KernelExplainer` as it provides a robust way to calculate SHAP values independently of the model. Due to the multi-class nature of our classification problem, we use `KernelExplainer` with a wrapper around `predict_proba` for the models that output probabilities, otherwise, we use the predict method. For a data point *i* and feature *j*, the SHAP value (φᵢ(j)) represents its contribution and can be mathematically approximated as:

# φᵢ(j) ≈ Σ ( [f(zᵢ + j) - f(zᵢ - j) ] )

# where f is the model's prediction function, zᵢ is the data point *i*, zᵢ+j represents a data point with feature j perturbed and zᵢ-j represents a data point where the feature j is not perturbed.  The sum is over all possible combinations of feature perturbations. The SHAP value for a data point and feature represents the difference of its prediction compared to the average of all predictions.

# Thirdly, we generate **partial dependence plots (PDPs)** to illustrate how each feature impacts the model's prediction across its range for each class. PDPs visualize the marginal effect of a feature on the predicted outcome, holding all other features constant. Mathematically, the partial dependence of feature *j* at value *x* is given by:

# PDPⱼ(x) = Eₓ[f(X)|Xⱼ = x]

# where Eₓ denotes the expectation, f(X) is the model's prediction function, and *Xⱼ* is the feature *j*. We calculate this over a grid of values for *x* to visualize the impact of this feature on prediction.

# Finally, for the models that have it, we extract coefficient information, providing directionality on the model decision. We also compute and store the feature importance information separately for static and dynamic features, allowing us to gain insights into each type of feature. Through these explainability tools, we aim to provide a thorough and reliable understanding of the factors driving the model's predictions, which can then be used to improve the models and inform decision-making.

import pandas as pd
import pickle
import numpy as np
import os
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import f1_score
from sklearn.inspection import PartialDependenceDisplay
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load saved variables
save_path = 'variables.pkl'
with open(save_path, 'rb') as f:
    loaded_variables = pickle.load(f)
best_classifiers = loaded_variables['best_classifiers']
X_test = loaded_variables['X_test']
y_test = loaded_variables['y_test']
dynamic_vars = loaded_variables['dynamic_vars']
static_vars = loaded_variables['static_vars']
train_test_label = loaded_variables['train_test_label']

expanded_dynamic_vars = []
for var in dynamic_vars:
    for stat in ['mean', 'std', 'min', 'max', 'kurtosis', 'skewness']:
        expanded_dynamic_vars.append(f'{var}_{stat}')

all_feature_names = expanded_dynamic_vars + static_vars

# ---------------------------------------------------
# 1. Feature Importance (current implementation + fixes)
# ---------------------------------------------------
print("Estimating feature importance...")
feature_importances = {}
feature_selection = {}  # Store feature selection
for name, clf in best_classifiers.items():
    print(f"Calculating feature importance for {name}...")
    if hasattr(clf, 'feature_importances_'):
        feature_importances[name] = clf.feature_importances_
        feature_selection[name] = all_feature_names
    elif hasattr(clf, 'coef_'):
        feature_importances[name] = clf.coef_[0]  # Keep sign information
        feature_selection[name] = all_feature_names
    else:
        # Use permutation importance as fallback
        def calculate_permutation_importance(clf, X, y, feature_names, sample_size=500):
            if len(X) > sample_size:
              idx = np.random.choice(len(X), sample_size, replace=False)
              X_sample, y_sample = X[idx], y[idx]
            else:
              X_sample, y_sample = X, y

            result = permutation_importance(clf, X_sample, y_sample, n_repeats=10, random_state=42, scoring='f1_weighted') # Using f1
            return pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean})

        perm_importance_df = calculate_permutation_importance(clf, X_test, y_test, all_feature_names, sample_size=500)
        feature_importances[name] = perm_importance_df.set_index('Feature')['Importance'].values
        feature_selection[name] = all_feature_names
        perm_importance_df.to_csv(os.path.join('results', f'permutation_importance_{name}.csv'), index=False)

# ---------------------------------------------------
# 2. SHAP (SHapley Additive exPlanations)
# ---------------------------------------------------
print("Calculating SHAP values...")
shap_values_dict = {}
background_size = 50  # You can change this parameter
for name, clf in best_classifiers.items():
    try:
        print(f"  Calculating SHAP values for {name}...")
        if hasattr(clf, 'predict_proba'): # Models with predict_proba
            # Sample or use K-Means to reduce background data
            background_data = shap.sample(X_test, background_size)  # Use random sampling
             # Create a wrapper to return only the probability of the positive class for the background data
            def predict_proba_wrapper(background_data):
              return clf.predict_proba(background_data)[:, 1]  # returns the probability of the second class
            explainer = shap.KernelExplainer(predict_proba_wrapper, background_data)
            shap_values = explainer.shap_values(X_test)
        elif hasattr(clf, 'predict'): # Models with predict only
            # Sample or use K-Means to reduce background data
            background_data = shap.sample(X_test, background_size)
            explainer = shap.KernelExplainer(clf.predict, background_data)
            shap_values = explainer.shap_values(X_test)
        else: # Fallback if none of the conditions match
            print(f"   {name} doesn't have a  predict_proba or predict method, skipping SHAP calculation")
            shap_values = None # if no SHAP values

        # Save SHAP values
        if shap_values is not None:
            with open(os.path.join('results', f'shap_values_{name}.pkl'), 'wb') as f:
               pickle.dump(shap_values, f)

            # Generate and save summary plot
            plt.figure()
            if isinstance(shap_values, list):  # Handle multiclass output
                for i, class_shap_values in enumerate(shap_values):
                    shap.summary_plot(class_shap_values, X_test, feature_names = all_feature_names, plot_type="bar",show=False)
                    plt.savefig(os.path.join('results', f'shap_summary_plot_{name}_class{i}.png'))
            else: # Handle other situations
                 shap.summary_plot(shap_values, X_test, feature_names = all_feature_names, plot_type="bar",show=False)
                 plt.savefig(os.path.join('results', f'shap_summary_plot_{name}.png'))
            plt.close() # Close figure

    except Exception as e:
        print(f"    SHAP calculation failed for {name}. Error: {e}")

# ---------------------------------------------------
# 3. Partial Dependence Plots (PDP)
# ---------------------------------------------------
print("Generating Partial Dependence Plots...")
for name, clf in best_classifiers.items():
    try:
        print(f"  Generating PDP for {name}...")
        if hasattr(clf, 'predict_proba'):
            for feature_idx, feature_name in enumerate(all_feature_names):
                for target_class in range(len(np.unique(y_test))): #iterate over the classes
                    fig, ax = plt.subplots(figsize=(8, 6))
                    PartialDependenceDisplay.from_estimator(clf, X_test, [feature_idx], target = target_class, feature_names = all_feature_names, ax = ax) # Plot PDP for class
                    plt.title(f"PDP for {feature_name} with {name} (Class {target_class})")
                    plt.savefig(os.path.join('results', f'pdp_{name}_{feature_name}_class{target_class}.png'))
                    plt.close()  # Close figure to avoid overlapping
        else:
            print(f"    Skipping PDP for {name} (does not support predict_proba)")
    except Exception as e:
        print(f"    PDP generation failed for {name}. Error: {e}")

# ---------------------------------------------------
# 4. Separate Static and Dynamic Importances
# ---------------------------------------------------
print("Separating static and dynamic importances...")
static_importance_df = pd.DataFrame()
dynamic_importance_df = pd.DataFrame()
num_stats_per_dynamic_var = 6
static_len = len(static_vars)
dynamic_len = len(dynamic_vars) * num_stats_per_dynamic_var

for name, importances in feature_importances.items():
    total_length = len(importances)
    print(f"  {name}: Total features = {total_length}, Static = {static_len}, Dynamic = {dynamic_len}")

    # Check if the classifier supports feature selection
    if  name in feature_selection and len(feature_selection[name]) == total_length:
        static_importance_df[name] = pd.Series(importances[:static_len], index=static_vars)
        dynamic_importance_df[name] = pd.Series(importances[static_len:], index=expanded_dynamic_vars)
    else:
        print(f"    Warning: Mismatch in feature lengths for {name}. Handling without selection...")
        static_importance_df[name] = pd.Series(importances[:static_len], index=static_vars)
        if total_length > static_len:
            dynamic_importance_df[name] = pd.Series(importances[static_len:], index=expanded_dynamic_vars[:total_length-static_len])
        else:
            dynamic_importance_df[name] = pd.Series([np.nan] * dynamic_len, index=expanded_dynamic_vars)

# Save the DataFrames to CSV files
static_importance_df.to_csv(os.path.join('results',
                            f'static_variable_importance_{train_test_label}.csv'), index_label='Static_Variable')
dynamic_importance_df.to_csv(os.path.join('results',
                             f'dynamic_variable_importance_{train_test_label}.csv'), index_label='Dynamic_Variable')

print(static_importance_df.head())
print(dynamic_importance_df.head())

print("Finished explainability analysis.")