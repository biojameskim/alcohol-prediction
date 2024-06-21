from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd

import process_conn_matrices as pcm

### I'm going to have to extract the upper triangler only first ###
"""
Steps: 
1. Extract FC/SC baseline matrices
  1a. Extract FC matrices for all subjects (both <18 and >=18)
  1b. Extract SC matrices for all subjects (from "tractography_subcortical" folder)
2. Load the FC and SC matrices and the target labels (y)
  2a. The y is group1 and group2 so I have to first identify individuals in group1 and group2 and then assign them 0 and 1
3. Extract them into upper triangular matrices
4. Look at the code below and see which to use/modify/take out.
"""

subjects_with_groups_df = pd.read_csv('data/csv/simple_view.csv') # simple_view.csv is a CSV file with columns: subject, group, cahalan

SC_matrices, subjects = pcm.load_and_flatten_conn_matrices('data/tractography_subcortical', False)
# FC_matrices, subjects = load_and_flatten_conn_matrices('../data/TODO', True)

# ******* Each subject has same y_SC so maybe make get_X_y return both X_SC and X_FC?
# or keeping it this way would ensure that each X has the correct y
X_SC, y_SC = pcm.get_X_y(SC_matrices, subjects, subjects_with_groups_df)
# X_FC, y_FC = get_X_y(FC_matrices, subjects, subjects_with_groups_df)

# FC matrix inputs (shape: n_samples x 11881)
X_FC = None
# Your binary target labels (0 or 1)
y_FC = None

# Train-test split
X_FC_train, X_FC_test, y_FC_train, y_FC_test = train_test_split(X_FC, y_FC, test_size=0.2, random_state=42)
X_SC_train, X_SC_test, y_SC_train, y_SC_test = train_test_split(X_SC, y_SC, test_size=0.2, random_state=42)

# Logistic Regression with Ridge Penalty
model_FC = LogisticRegression(penalty='l2', C=1.0)  # C is the inverse of λ
model_SC = LogisticRegression(penalty='l2', C=1.0)

# Train the models
model_FC.fit(X_FC_train, y_FC_train)
model_SC.fit(X_SC_train, y_SC_train)

# Predict probabilities
proba_fc = model_FC.predict_proba(X_FC_test)[:, 1]  # Probability of class 1
proba_sc = model_SC.predict_proba(X_SC_test)[:, 1]  # Probability of class 1


### Ensemble Models ###

# Simple averaging
ensemble_proba_avg = (proba_fc + proba_sc) / 2

# Weighted sum
w1 = 0.5
w2 = 0.5
ensemble_proba_weighted = w1 * proba_fc + w2 * proba_sc

# Convert probabilities to binary predictions
ensemble_predictions_avg = (ensemble_proba_avg >= 0.5).astype(int)
ensemble_predictions_weighted = (ensemble_proba_weighted >= 0.5).astype(int)

# Evaluate the ensemble models
accuracy_avg = accuracy_score(y_test, ensemble_predictions_avg)
accuracy_weighted = accuracy_score(y_test, ensemble_predictions_weighted)
roc_auc_avg = roc_auc_score(y_test, ensemble_proba_avg)
roc_auc_weighted = roc_auc_score(y_test, ensemble_proba_weighted)

print(f"Ensemble (Averaging) Accuracy: {accuracy_avg}, ROC AUC: {roc_auc_avg}")
print(f"Ensemble (Weighted) Accuracy: {accuracy_weighted}, ROC AUC: {roc_auc_weighted}")


