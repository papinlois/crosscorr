#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:26:34 2024

Plot the original detections and the highest cc coeff of the new ones. The code
is written for the same number of detections stacked as for the picker and the 
only thing to change is the template we want to plot for.

NB: Made for local use.

@author: lpapin

As of 01/07/24.
"""

#%% Detections
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import crosscorr_tools

# Load and preprocess detections
detections = pd.read_csv('/Users/lpapin/Desktop/SSE_2005/crosscorr/aug_PO_2/yes/output_aug_PO_yes.txt')
detections = detections.sort_values(['starttime'])
print(detections)

# Filter specific template
templ = 3 ###
detections = detections[detections['template'] == templ]
print(detections)

# Get the original event with the highest cc coefficient
cc_coeffmax = np.max(detections['coeff'])
original_event = detections[detections['coeff'] == cc_coeffmax]
print(original_event)

# Filter by time period
detections['starttime'] = pd.to_datetime(detections['starttime'])
startdate = datetime.strptime(detections['starttime'].min().strftime("%Y%m%d"), "%Y%m%d")
enddate = datetime.strptime(detections['starttime'].max().strftime("%Y%m%d"), "%Y%m%d")
starttime = datetime.combine(startdate, datetime.strptime("00:00:00", "%H:%M:%S").time())
endtime = datetime.combine(enddate, datetime.strptime("23:59:59", "%H:%M:%S").time())
detections = detections[(detections['starttime'] >= starttime) & (detections['starttime'] < endtime)]
print(detections)

#%% Streams
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the base directory and network
base_dir = "/Users/lpapin/Documents/phd/"
from network_configurations import network_config  # for now for PO

# Get the dates of interest
dates_of_interest = []
current_date = startdate
while current_date <= enddate:
    dates_of_interest.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)
lastday = dates_of_interest[-1]  # For filenames

# Get the streams
st, pairs = crosscorr_tools.get_traces(network_config, dates_of_interest, base_dir)

# Define frequency range, sampling rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1 / sampling_rate
win_size = 10
st = crosscorr_tools.process_data(st, startdate, enddate + timedelta(days=1), sampling_rate, freqmin, freqmax)
print(st.__str__(extended=True))

# =============================================================================
# #%% Plotting
# 
# # Define the number of detections to stack
# N_values = [1, 2, 5, 10, 20, 50, 100]
# 
# # Iterate over each N value
# for N in N_values:
#     detections = detections.sort_values(['coeff'], ascending=False).reset_index(drop=True)
#     detections.index.name = 'Index'
# 
#     # Optionally exclude the highest coefficient detection if it matches cc_coeffmax
#     if detections['coeff'][0] == cc_coeffmax:
#         detections = detections[1:].reset_index(drop=True)
# 
#     # Adjust the number of subplots based on N
#     plot_detections = min(N, 4)
#     total_subplots = plot_detections + 2
# 
#     # Iterate over station picks
#     station_picks = ['KLNB', 'SILB', 'SSIB', 'TSJB', 'TWKB']
#     folder_path = '/Users/lpapin/Desktop/SSE_2005/picker/full/Detections_S_PO_stack'
# 
#     # Adjust the figure creation accordingly
#     for tr in st:
#         fig, axs = plt.subplots(total_subplots, figsize=(10, 20))
#         fig.suptitle(f'Template {templ} for {tr.id}', fontsize=16)
# 
#         # Plot original event
#         original_event_time = original_event['starttime'].iloc[0]
#         start_templ = UTCDateTime(original_event_time)
#         end_templ = start_templ + timedelta(seconds=win_size)
#         original_event_data = tr.copy().trim(starttime=start_templ, endtime=end_templ)
#         axs[0].plot(original_event_data.times(), original_event_data.data, color='orange', linewidth=2)
#         axs[0].set_title('Original Event: {}'.format(start_templ))
#         axs[0].grid(True)
# 
#         # Plot detections
#         stacked_data = np.zeros(len(original_event_data.data))
#         for i, detection in detections[:plot_detections].iterrows():
#             detection_time = detection['starttime']
#             start_det = UTCDateTime(detection_time)
#             end_det = start_det + timedelta(seconds=win_size)
#             detection_data = tr.copy().trim(starttime=start_det, endtime=end_det)
#             axs[i + 1].plot(detection_data.times(), detection_data.data, linewidth=1)
#             axs[i + 1].set_title('Detection {}: {} - cc={:.2f}'.format(i + 1, start_det, detection['coeff']))
#             axs[i + 1].grid(True)
#             stacked_data += detection_data.data
#             
#         # Stack all N detections for the last subplot
#         full_stacked_data = np.zeros(len(original_event_data.data))
#         for _, detection in detections[:N].iterrows():
#             detection_time = detection['starttime']
#             start_det = UTCDateTime(detection_time)
#             end_det = start_det + timedelta(seconds=win_size)
#             detection_data = tr.copy().trim(starttime=start_det, endtime=end_det)
#             full_stacked_data += detection_data.data
# 
#         # Plot stacked waveform for the first N detections in the last subplot
#         axs[total_subplots - 1].plot(original_event_data.times(), full_stacked_data, color='green', linewidth=1)
#         axs[total_subplots - 1].set_title(f'Stacked {N} Detections with S-wave Computed Arrival Time')
#         axs[total_subplots - 1].grid(True)
#         
#         # Add vertical red line if N and template match
#         picks = pd.read_csv(f'{folder_path}/cut_daily_PO.{tr.stats.station}.csv')
#         mask = (picks['nb_stack'] == N) & (picks['lfe_family'] == templ)
#         if mask.any():
#             axs[total_subplots - 1].axvline(x=picks['S_times'][mask].iloc[0], color='red', linestyle='--')
# 
#         plt.tight_layout()
#         plt.savefig(f'{folder_path[:30]}/stacks/stacked_detections_{tr.id}_templ{templ}_N={N}.png', dpi=300)
#         plt.close()
# 
# =============================================================================
#%% Plotting stacked detections

# Define the number of detections to stack
N_values = [2, 5, 10, 20, 50, 100]  # 1=original event

# Iterate over each station pick
station_picks = ['KLNB', 'SILB', 'SSIB', 'TSJB', 'TWKB']
folder_path = '/Users/lpapin/Desktop/SSE_2005/picker/full/Detections_S_PO_stack'

for tr in st:
    # Create a figure with the required number of subplots
    fig, axs = plt.subplots(len(N_values) + 1, figsize=(10, 20))
    fig.suptitle(f'Template {templ} for {tr.id}', fontsize=16)

    # Plot the original event
    original_event_time = original_event['starttime'].iloc[0]
    start_templ = UTCDateTime(original_event_time)
    end_templ = start_templ + timedelta(seconds=win_size)
    original_event_data = tr.copy().trim(starttime=start_templ, endtime=end_templ)
    axs[0].plot(original_event_data.times(), original_event_data.data, color='orange', linewidth=2)
    axs[0].set_title('Original Event: {}'.format(start_templ))
    axs[0].grid(True)

    # Iterate over each N value
    for idx, N in enumerate(N_values):
        detections = detections.sort_values(['coeff'], ascending=False).reset_index(drop=True)
        detections.index.name = 'Index'

        # Optionally exclude the highest coefficient detection if it matches cc_coeffmax
        if detections['coeff'][0] == cc_coeffmax:
            detections = detections[1:].reset_index(drop=True)

        # Stack all N detections
        full_stacked_data = np.zeros(len(original_event_data.data))
        for _, detection in detections[:N].iterrows():
            detection_time = detection['starttime']
            start_det = UTCDateTime(detection_time)
            end_det = start_det + timedelta(seconds=win_size)
            detection_data = tr.copy().trim(starttime=start_det, endtime=end_det)
            full_stacked_data += detection_data.data

        # Plot stacked waveform for the current N detections
        axs[idx + 1].plot(original_event_data.times(), full_stacked_data, color='green', linewidth=1)
        axs[idx + 1].set_title(f'Stacked {N} Detections with S-wave Computed Arrival Time')
        axs[idx + 1].grid(True)

        # Add vertical red line if N and template match
        picks = pd.read_csv(f'{folder_path}/cut_daily_PO.{tr.stats.station}.csv')
        mask = (picks['nb_stack'] == N) & (picks['lfe_family'] == templ)
        if mask.any():
            axs[idx + 1].axvline(x=picks['S_times'][mask].iloc[0], color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{folder_path[:30]}/stacks/stacked_detections_{tr.id}_templ{templ}_Ns.png', dpi=300)
    plt.close()

