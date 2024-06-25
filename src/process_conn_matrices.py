import os
import scipy.io
import numpy as np
import pandas as pd

def load_and_flatten_SC_matrices(directory):
    """
    Load and flatten the SC matrices from the given directory.
    Returns: [SC_matrices]: A 2D array of flattened SC matrices
             [subject_ids]: An array of matching (in order of the SC matrices) subject IDs
    """
    SC_matrices = []
    subject_ids = []
    
    for filename in os.listdir(directory):
        if filename.endswith('baseline.mat'): # Only extract baseline matrices
            file_path = os.path.join(directory, filename)
            mat_contents = scipy.io.loadmat(file_path)
            SC_matrix = mat_contents['matrix'] # Extract the connectivity matrix from the .mat file

            diagonal_indices = np.triu_indices(SC_matrix.shape[0], 1) 
            SC_matrix = SC_matrix[diagonal_indices] # Extract the upper triangular part of the matrix (excluding the diagonal)
            
            flattened_matrix = SC_matrix.flatten() # Flatten the matrix into a 1D array
            
            # Store the flattened matrix and subject id
            SC_matrices.append(flattened_matrix)
            subject_id = "NCANDA_" + filename.split('_')[1]  # Extract subject number, e.g., 'NCANDA_S00033'
            subject_ids.append(subject_id)
    
    return np.array(SC_matrices), np.array(subject_ids)

def load_FC_matrices_and_subject_ids(path_to_FC, path_to_FCgsr, path_to_demography):
    """
    Load the FC matrices (with and without gsr) and extract the subject IDs.
    Returns: [baseline_FC_matrices]: A 2D array of baseline FC matrices
             [baseline_FCgsr_matrices]: A 2D array of baseline FCgsr matrices
             [subject_ids]: An array of matching (in order of the baseline FC matrices) subject IDs
    """
    FC_mat_contents = scipy.io.loadmat(path_to_FC)
    FC_cell_array = FC_mat_contents['FC']
    FC_matrices = FC_cell_array[0] # This is an array of FC matrices for all subjects (contains duplicate subjects based on visit)

    # Same process for FCgsr matrices
    FCgsr_mat_contents = scipy.io.loadmat(path_to_FCgsr)
    FCgsr_cell_array = FCgsr_mat_contents['FC']
    FCgsr_matrices = FCgsr_cell_array[0]

    demos = pd.read_csv(path_to_demography)
    baseline_subjects = demos[demos['visit'] == 'baseline'] # Filter for baseline subjects
    baseline_indices = baseline_subjects.index

    baseline_FC_matrices = FC_matrices[baseline_indices] # extract FC matrices for just the baseline
    baseline_FCgsr_matrices = FCgsr_matrices[baseline_indices] # extract FCgsr matrices for just the baseline

    # Extract matching subject ids, e.g., 'NCANDA_S00033'
    subject_ids = np.array(baseline_subjects['subject'])

    return baseline_FC_matrices, baseline_FCgsr_matrices, subject_ids

def flatten_FC_matrices(baseline_FC_matrices):
    """
    Flatten the FC matrices and return a 2D array of flattened matrices.
    """
    flattened_FC_matrices = []
    for FC_matrix in baseline_FC_matrices:
        diagonal_indices = np.triu_indices(FC_matrix.shape[0], 1) 
        FC_matrix = FC_matrix[diagonal_indices] # Extract the upper triangular part of the matrix (excluding the diagonal)
        flattened_matrix = FC_matrix.flatten() # Flatten the matrix into a 1D array
        # Store the flattened matrix and subject id
        flattened_FC_matrices.append(flattened_matrix)

    # Convert list to NumPy arrays
    flattened_FC_matrices = np.array(flattened_FC_matrices)

    return flattened_FC_matrices

def get_X_y(conn_matrices, subject_ids, subjects_with_groups_df):
    """
    Prepare the feature matrix X and target labels (groups) y for the logistic regression model.
    [conn_matrices] is a 2D array of flattened FC/SC matrices for all subjects
    [subjects] is a 1D array of subject IDs (e.g., "NCANDA_S00033")
    [subjects_with_groups_df] is a DataFrame with each subject and their group label
    Returns: A tuple of (X, y) where X is a 2D array and y is a 1D array
    """
    conn_matrices_df = pd.DataFrame(conn_matrices)
    conn_matrices_df['subject'] = subject_ids

    conn_matrices_with_groups = conn_matrices_df.merge(subjects_with_groups_df, on='subject') # Merge with the group labels
    conn_matrices_with_groups = conn_matrices_with_groups.drop_duplicates(subset=['subject']) # drop all duplicates

    X = conn_matrices_with_groups.drop(columns=['subject', 'group', 'cahalan']).values  # Keep just the matrices (Based on structure of simple_view.csv)
    y = conn_matrices_with_groups['group'].values  # Group labels

    return X, y

if __name__ == "__main__":
    subjects_with_groups_df = pd.read_csv('data/csv/simple_view.csv')

    # SC matrices
    SC_matrices, subject_ids = load_and_flatten_SC_matrices('data/tractography_subcortical (SC)')
    X_SC, y_SC = get_X_y(SC_matrices, subject_ids, subjects_with_groups_df)
    np.save('data/X_SC.npy', X_SC)
    np.save('data/y_SC.npy', y_SC)

    # FC matrices
    baseline_FC_matrices, baseline_FCgsr_matrices, subject_ids = load_FC_matrices_and_subject_ids(
    path_to_FC='data/FC/NCANDA_FC.mat', 
    path_to_FCgsr='data/FC/NCANDA_FCgsr.mat',
    path_to_demography='data/FC/NCANDA_demos.csv'
    )

    FC_matrices = flatten_FC_matrices(baseline_FC_matrices)
    X_FC, y_FC = get_X_y(FC_matrices, subject_ids, subjects_with_groups_df)

    # FCgsr
    FCgsr_matrices = flatten_FC_matrices(baseline_FCgsr_matrices)
    X_FCgsr, y_FCgsr = get_X_y(FCgsr_matrices, subject_ids, subjects_with_groups_df)

    np.save('data/X_FC.npy', X_FC)
    np.save('data/y_FC.npy', y_FC)
    np.save('data/X_FCgsr.npy', X_FCgsr)
    np.save('data/y_FCgsr.npy', y_FCgsr)








