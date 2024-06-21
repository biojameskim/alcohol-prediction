import os
import scipy.io
import numpy as np
import pandas as pd

def load_and_flatten_conn_matrices(directory):
    """
    Load and flatten the connectivity matrices from the given directory.
    Returns: A tuple of (conn_matrices, subjects) where conn_matrices and subjects are arrays
    """
    conn_matrices = []
    subjects = []
    
    for filename in os.listdir(directory):
        if filename.endswith('baseline.mat'): # Only extract baseline matrices
            file_path = os.path.join(directory, filename)
            mat_contents = scipy.io.loadmat(file_path)
            conn_matrix = mat_contents['matrix'] # Extract the connectivity matrix from the .mat file

            diagonal_indices = np.triu_indices(conn_matrix.shape[0], 1) 
            conn_matrix = conn_matrix[diagonal_indices] # Extract the upper triangular part of the matrix (excluding the diagonal)
            
            flattened_matrix = conn_matrix.flatten() # Flatten the matrix into a 1D array
            
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






