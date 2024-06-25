from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump
import numpy as np

def create_model_and_metrics(X, y, num_splits=5, num_repeats=100):
  """
  Perform nested cross-validation for logistic regression on connectivity matrices.
  Returns various metrics and the final model pipeline.
  """
  C_values = np.logspace(-4, 4, 15)  # 15 values from 10^-4 to 10^4

  # Precompute and reuse standard scaling
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Preallocating metrics arrays
  balanced_accuracies = np.empty(num_repeats * 5)
  roc_aucs = np.empty(num_repeats * 5)
  accuracies = np.empty(num_repeats * num_splits)
  idx = 0

  for _ in range(num_repeats):
    outer_kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=None)
    for train_index, test_index in outer_kf.split(X, y):
      X_train, X_test = X_scaled[train_index], X_scaled[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Nested CV for hyperparameter tuning
      inner_kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=None)

      model= LogisticRegressionCV(
          penalty='l2', 
          Cs=C_values, 
          cv=inner_kf, 
          random_state=42, 
          max_iter=100,
          n_jobs=-1 # Use all available CPU cores
      )

      # Train model on outer fold training data
      model.fit(X_train, y_train)

      # Predict and evaluate on outer fold test data
      y_pred = model.predict(X_test)
      y_prob = model.predict_proba(X_test)[:, 1] # Probability of class 1

      # Calculate standard accuracy
      accuracy = accuracy_score(y_test, y_pred)
      accuracies[idx] = accuracy

      # Calculate balanced accuracy
      bal_acc = balanced_accuracy_score(y_test, y_pred)
      balanced_accuracies[idx] = bal_acc

      # Calculate ROC AUC score
      roc_auc = roc_auc_score(y_test, y_prob)
      roc_aucs[idx] = roc_auc

      idx += 1

  results = {
    "accuracies": accuracies,
    "mean_accuracy": np.mean(accuracies),
    "std_accuracy": np.std(accuracies),
    "balanced_accuracies": balanced_accuracies,
    "mean_balanced_accuracy": np.mean(balanced_accuracies),
    "std_balanced_accuracy": np.std(balanced_accuracies),
    "roc_aucs": roc_aucs,
    "mean_roc_auc": np.mean(roc_aucs),
    "std_roc_auc": np.std(roc_aucs),
    "pipeline": make_pipeline(scaler, model)  # Return the final pipeline
    }
  
  return results

def logreg_conn_matrices(matrix_type, num_splits, num_repeats):
  """
  Perform logistic regression on [matrix_type] matrices.
  [matrix_type] should be a string of either 'SC', 'FC', or 'FCgsr'.
  Prints results to a report file in the results/[matrix_type] directory.
  """
  X = np.load(f'data/X_{matrix_type}.npy')
  y = np.load(f'data/y_{matrix_type}.npy')

  results = create_model_and_metrics(X=X, y=y, num_splits=num_splits, num_repeats=num_repeats)

  # Save the model
  dump(results['pipeline'], f'models/logreg_{matrix_type}_model.joblib')

  # Save the results
  np.save(f'results/{matrix_type}/logreg_{matrix_type}_accuracies.npy', results['accuracies'])
  np.save(f'results/{matrix_type}/logreg_{matrix_type}_balanced_accuracies.npy', results['balanced_accuracies'])
  np.save(f'results/{matrix_type}/logreg_{matrix_type}_roc_aucs.npy', results['roc_aucs'])

  # Put the results in a report
  report_lines = [
    f"Logistic Regression Results for {matrix_type} matrix:",
    f"Number of Splits: {num_splits}",
    f"Number of Repeats: {num_repeats}\n",
    f"Mean Accuracy: {results['mean_accuracy']}",
    f"Standard Deviation of Accuracy: {results['std_accuracy']}",
    f"Mean Balanced Accuracy: {results['mean_balanced_accuracy']}",
    f"Standard Deviation of Balanced Accuracy: {results['std_balanced_accuracy']}",
    f"Mean ROC AUC: {results['mean_roc_auc']}",
    f"Standard Deviation of ROC AUC: {results['std_roc_auc']}"
  ]

  # Write the results to a report file
  with open(f'results/{matrix_type}/report_logreg_{matrix_type}.txt', 'w') as report_file:
      report_file.write("\n".join(report_lines))

  print(f"Printed results to results/{matrix_type}/report_logreg_{matrix_type}.txt")

if __name__ == "__main__":
  logreg_conn_matrices(matrix_type='SC', num_splits=5, num_repeats=1)
  logreg_conn_matrices(matrix_type='FC', num_splits=5, num_repeats=10)
  logreg_conn_matrices(matrix_type='FCgsr', num_splits=5, num_repeats=1)
