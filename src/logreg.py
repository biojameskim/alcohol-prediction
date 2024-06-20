from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

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

# FC matrix inputs (shape: n_samples x 11881)
X_fc = None
# Your binary target labels (0 or 1)
y = None
# SC matrix inputs (shape: n_samples x 8100)
X_sc = None

# Train-test split
X_fc_train, X_fc_test, y_train, y_test = train_test_split(X_fc, y, test_size=0.2, random_state=42)
X_sc_train, X_sc_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)

# Logistic Regression with Ridge Penalty
model_fc = LogisticRegression(penalty='l2', C=1.0)  # C is the inverse of λ
model_sc = LogisticRegression(penalty='l2', C=1.0)

# Train the models
model_fc.fit(X_fc_train, y_train)
model_sc.fit(X_sc_train, y_train)

# Predict probabilities
proba_fc = model_fc.predict_proba(X_fc_test)[:, 1]  # Probability of class 1
proba_sc = model_sc.predict_proba(X_sc_test)[:, 1]  # Probability of class 1


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


