#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:18:28 2024

Compute some maths on the output file full of the new events.

NB: Made for local use.

@author: lpapin

As of 01/07/24.
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# Function to plot and save figure
def plot_and_save(picks, station_name):
    templates = picks['lfe_family'].unique()
    templates.sort()
    colors = plt.cm.get_cmap('tab20', len(templates))
    templ_colors = {template: colors(i) for i, template in enumerate(templates)}
    picks['color'] = picks['lfe_family'].map(templ_colors)
    # min_size, max_size = picks['nb_stack'].min(), picks['nb_stack'].max()
    # mean_cc_normalized = (picks['mean_cc'] - picks['mean_cc'].min()) / (picks['mean_cc'].max() - picks['mean_cc'].min())
    # dot_sizes = mean_cc_normalized * (max_size - min_size) + min_size

    unique_nb_stack = np.sort(picks['nb_stack'].unique())
    normalized_ticks = np.linspace(0, 1, len(unique_nb_stack))
    nb_stack_normalized = dict(zip(unique_nb_stack, normalized_ticks))
    picks['nb_stack_normalized'] = picks['nb_stack'].map(nb_stack_normalized)

    plt.figure(figsize=(7,7))
    for template in templates:
        subset = picks[picks['lfe_family'] == template]
        plt.scatter(subset['S_times'], subset['nb_stack_normalized'], c=subset['color'], s=subset['mean_cc']*100, marker='o')
        plt.plot(subset['S_times'], subset['nb_stack_normalized'], color=subset['color'].iloc[0], linestyle='-', linewidth=0.8)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=str(template)) for template, color in templ_colors.items()]
    plt.legend(title='Template', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'Effect of stacking on S-wave arrival times \n Station: {station_name}', fontsize=16)
    plt.xlabel('Arrival times of S-wave (s)', fontsize=14)
    plt.ylabel('Number of detections stacked', fontsize=14)
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.5)
    plt.yticks(ticks=normalized_ticks, labels=unique_nb_stack)
    plt.tight_layout()
    plt.savefig(f'{station_name}_plot.png', dpi=300)
    plt.close()

# Iterate over station picks
station_picks = ['KLNB', 'SILB', 'SSIB', 'TSJB', 'TWKB']
folder_path = '/Users/lpapin/Desktop/SSE_2005/picker/full/Detections_S_PO_stack'
# path='/Users/lpapin/Downloads/cut_daily_PO.KLNB.csv'

for station in station_picks:
    csv_file = glob.glob(f'{folder_path}/cut_daily_PO.{station}.csv')[0]
    picks = pd.read_csv(csv_file)
    # picks=pd.read_csv(path)
    plot_and_save(picks, station)
