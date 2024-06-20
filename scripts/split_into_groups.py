#!/usr/bin/env python3

"""
Name: split_into_groups.py
Purpose: Split the subjects into two groups based on their 'cahalan' status.
    - Group 0 --> No subsequent time points with cahalan 'heavy' or 'heavy_with_binging"
    - Group 1 --> At least one time point with cahalan 'heavy' or 'heavy_with_binging"
"""

import pandas as pd

df = pd.read_csv('data/csv/cahalan_plus_drugs.csv')

# Filter for subjects that are "control" on visit 0
control_on_visit_0 = df[(df['visit'] == 0) & (df['cahalan'] == 'control')]

# Get unique subject IDs for those controls on visit 0
control_subjects = control_on_visit_0['subject'].unique()

# Initialize an empty set to hold subjects who later become "heavy" or "heavy_with_binging"
heavy_subjects = set()

# Loop through each control subject and check their subsequent visits
for subject in control_subjects:
    subject_data = df[df['subject'] == subject]
    if any(subject_data['cahalan'].isin(['heavy', 'heavy_with_binging'])):
        heavy_subjects.add(subject)

# Convert the set to a list
heavy_subjects = list(heavy_subjects)

# Get all unique subjects
all_subjects = df['subject'].unique()

# Determine subjects in the second group (everyone else)
no_heavy_subjects = list(set(all_subjects) - set(heavy_subjects))

# Update the 'group' column based on the subject's status
# Create a new column 'group' and initialize with '0'
df['group'] = 0

# Set the value to '1' for subjects in the heavy group
df.loc[df['subject'].isin(heavy_subjects), 'group'] = 1

# Save to a new CSV file
df.to_csv('data/csv/cahalan_plus_drugs_with_groups.csv', index=False)

# Make a simple view of the data with just subject, cahalan, and group
simple_view = df[['subject', 'cahalan', 'group']]
simple_view.to_csv('data/csv/simple_view.csv', index=False)