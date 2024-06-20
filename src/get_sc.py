import os
import scipy.io
import numpy as np
import pandas as pd

def load_and_flatten_conn_matrices(directory, is_FC):
    """
    Load and flatten the connectivity matrices from the given directory.
    If is_FC is True, extract the upper triangular part of the matrix (excluding the diagonal).
    Returns: A tuple of (conn_matrices, subjects) where conn_matrices and subjects are arrays
    """
    conn_matrices = []
    subjects = []
    
    for filename in os.listdir(directory):
        if filename.endswith('baseline.mat'): # Only extract baseline matrices
            file_path = os.path.join(directory, filename)
            mat_contents = scipy.io.loadmat(file_path)
            conn_matrix = mat_contents['matrix'] # Extract the connectivity matrix from the .mat file

            if is_FC:
              diagonal_indices = np.triu_indices(conn_matrix.shape[0], 1)
              conn_matrix = conn_matrix[diagonal_indices]
            
            # Flatten the matrix into a 1D array
            flattened_matrix = conn_matrix.flatten()
            
            # Store the flattened matrix and subject id
            conn_matrices.append(flattened_matrix)
            subject_id = "NCANDA_" + filename.split('_')[1]  # Extract subject number, e.g., 'NCANDA_S00033'
            subjects.append(subject_id)
    
    return np.array(conn_matrices), np.array(subjects)

def get_X_y(conn_matrices, subjects, subjects_with_groups_df):
    """
    Prepare the feature matrix X and target labels (groups) y for the logistic regression model.
    [conn_matrices] is a 2D array of flattened FC/SC matrices for all subjects
    [subjects] is a 1D array of subject IDs (e.g., "NCANDA_S00033")
    [subjects_with_groups_df] is a DataFrame with each subject and their group label
    Returns: A tuple of (X, y) where X is a 2D array and y is a 1D array
    """
    conn_matrices_df = pd.DataFrame(conn_matrices)
    conn_matrices_df['subject'] = subjects

    conn_matrices_with_groups = conn_matrices_df.merge(subjects_with_groups_df, on='subject') # Merge with the group labels
    conn_matrices_with_groups = conn_matrices_with_groups.drop_duplicates(subset=['subject']) # drop all duplicates

    X = conn_matrices_with_groups.drop(columns=['subject', 'group', 'cahalan']).values  # Keep just the matrices (Based on structure of simple_view.csv)
    y = conn_matrices_with_groups['group'].values  # Group labels
    
    return X, y


if __name__ == "__main__":
  subjects_with_groups_df = pd.read_csv('data/csv/simple_view.csv') # simple_view.csv is a CSV file with columns: subject, group, cahalan

  SC_matrices, subjects = load_and_flatten_conn_matrices('data/tractography_subcortical', False)
  # FC_matrices, subjects = load_and_flatten_conn_matrices('../data/TODO', True)

  # ******* Each subject has same y_SC so maybe make get_X_y return both X_SC and X_FC?
  # or keeping it this way would ensure that each X has the correct y
  X_SC, y_SC = get_X_y(SC_matrices, subjects, subjects_with_groups_df)
  # X_FC, y_FC = get_X_y(FC_matrices, subjects, subjects_with_groups_df)

  print(X_SC.shape, y_SC.shape)





