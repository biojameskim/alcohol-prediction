#!/usr/bin/env python3

"""
Name: sort_hist_scores.py
Purpose: Sort the subjects based on their hist_g2_value and region_label.

Extra Notes:
- The columns in [histg2_mean_tzo116plus_cortex.csv] are <region_label> <region_name> <hist_g2_value>
- The "hist_g2" values are sign-flipped, so if you want "low level->high level", you would want the large positive values (red/yellow regions) to go first
- Subcortex regions were assigned a value of -0.5 (lower than the lowest value in the cortex) in the csv file

So in this case, we sign-flipped the values, so those values would go from about -0.37 to +0.39. (Subcortex regions are +0.5)
Then after sorting, the indices that would come out for reordering would be "low level->high level" (i.e., [subcortex regions] [yellow->red regions] [dark blue->light blue regions])
"""

import pandas as pd

df = pd.read_csv("/content/histg2_mean_tzo116plus_cortex.csv")

df['hist_g2_value'] = df['hist_g2_value'] * -1 # Reverse the sign

df['hierarchy_score'] = df['region_label'] * df['hist_g2_value']
sorted_df = df.sort_values(by=['hist_g2_value'])

sorted_df.to_csv('sorted_hierarchy_histg2.csv', index=False)
df.to_csv('hierarchy_histg2.csv', index=False)