#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:29:47 2024

@author: lpapin
"""

import pandas as pd
# from datetime import datetime
import matplotlib.pyplot as plt

# Load and process the templates
templates = pd.read_csv('./EQloc_001_0.1_3_S.csv', index_col=0)
templates = templates[(templates['residual'] < 0.1)]
templates['OT'] = pd.to_datetime(templates['OT'])  # Formatting 'OT' column as datetime
templates = templates.drop(columns=['residual', 'dt'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'

# Get the years of the templates
templates['Year'] = templates['OT'].dt.year
years = templates['Year'].unique()

# Initialize a list to store the statistics for each year
yearly_stats = []
for year in years:
    yearly_templates = templates[templates['Year'] == year]

    # Retrieve information about the templates for the current year
    first_template_date = yearly_templates['OT'].min()
    last_template_date = yearly_templates['OT'].max()
    total_templates = yearly_templates.shape[0]

    # Calculate the median and mean for the 'depth' column
    max_depth_value = yearly_templates['depth'].min()
    median_depth = yearly_templates['depth'].median()
    mean_depth = yearly_templates['depth'].mean()

    # Calculate the median and mean for the 'N' column
    max_N_value = yearly_templates['N'].max()
    median_N = yearly_templates['N'].median()
    mean_N = yearly_templates['N'].mean()

    # Append the statistics to the list
    yearly_stats.append({
        'Year': year,
        'First Template Date': first_template_date,
        'Last Template Date': last_template_date,
        'Total Templates': total_templates,
        'Max Depth': max_depth_value,
        'Median Depth': round(median_depth, 1),
        'Mean Depth': round(mean_depth, 1),
        'Max N': int(max_N_value),
        'Median N': int(median_N),
        'Mean N': int(mean_N)
    })

    # Plotting the histogram of the depth values for every 10km
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(yearly_templates['depth'], bins=range(-60, 11, 10), edgecolor='black')
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(),
                 str(int(n[i])), ha='center', va='bottom')
    plt.title(f'Histogram of Depth Values (every 10km) for Year {year}')
    plt.xlabel('Depth (km)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Create the csv file with the stats
yearly_stats_df = pd.DataFrame(yearly_stats)
yearly_stats_df.set_index('Year', inplace=True)
yearly_stats_df.to_csv('yearly_templates_stats.csv')
print(yearly_stats_df)
