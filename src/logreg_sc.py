from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

from process_conn_matrices import load_and_flatten_conn_matrices, get_X_y

subjects_with_groups_df = pd.read_csv('data/csv/simple_view.csv') # simple_view.csv is a CSV file with columns: subject, group, cahalan
SC_matrices, subjects = load_and_flatten_conn_matrices('data/tractography_subcortical')
X_SC, y_SC = get_X_y(SC_matrices, subjects, subjects_with_groups_df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_SC, y_SC, test_size=0.2, random_state=42)

C_values = np.logspace(-4, 4, 30)  # 30 values from 10^-4 to 10^4
# Define a logistic regression model wrapped in a pipeline with standard scaling
pipeline = make_pipeline(StandardScaler(), LogisticRegressionCV(penalty='l2', Cs=C_values, cv=5, random_state=42, max_iter=100))

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the testing set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(f"Best C: {pipeline.named_steps['logisticregressioncv'].C_}")
print(f"Score: {pipeline.score(X_test, y_test)}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))