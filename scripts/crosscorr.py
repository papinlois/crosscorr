# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

This module provides functions for seismic data processing, visualization, and analysis.
The functions cover loading and preprocessing data, plotting seismic traces, creating
summed traces around detected events, and more.

This version use the process_streams function, which examines a file to find 
out which stations performed the detection that we are utilizing as a template. 
Thus, we use such stations as templates at first, and then add other stations 
as new detections are made. (made for Tim's catalog)

As of 01/11/24.
"""

import os
import time
from datetime import datetime, timedelta
# import warnings
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from obspy import UTCDateTime
from scipy.signal import find_peaks
# from eqcorrscan.utils import findpeaks
# from network_configurations import network_config
import autocorr_tools
import crosscorr_tools

# import matplotlib
# matplotlib.use('Agg')

# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
folder = "tim"

# Start timer
startscript = time.time()

# Define the network configurations # or network_configurations.py with import
network_config = {
    'CN1': {
        'stations': ['LZB', 'PGC', 'SNB'],#, 'NLLB'
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'CN2': {
        'stations': ['YOUB', 'PFB'],#
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    # 'PB': {
    #     'stations': ['B001', 'B009', 'B011', 'B010', 'B926'],#
    #     'channels': ['EH1', 'EH2', 'EHZ'],
    #     'filename_pattern': '{station}.PB.{year}.{julian_day}'
    # },
    # 'C8': {
    #     'stations': ['MGCB', 'JRBC'],#, 'PHYB', 'SHVB', 'LCBC', 'GLBC', 'TWBB'], #
    #     'channels': ['HHN', 'HHE', 'HHZ'],
    #     'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    # },
    # 'PO': {
    #     'stations': ['SILB', 'SSIB', 'KLNB', 'TSJB'], #, 'TWKB'
    #     'channels': ['HHN', 'HHE', 'HHZ'],
    #     'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
    # }
}

# Days of data
startdate = datetime.strptime("20100504", "%Y%m%d")
enddate = datetime.strptime("20100520", "%Y%m%d")
date_of_interests = []
current_date = startdate
while current_date <= enddate:
    date_of_interests.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)
lastday=date_of_interests[-1] #For filenames

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 12

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax)

# List of stations/channels to analyze
pairs=[]
for tr in st:
    iid = tr.id
    identifier = iid[3:]
    pairs.append(identifier)

# Remove the bad data
tr_remove = ['PGC..BHE', 'SNB..BHE', 'B010..EH2', 'B010..EHZ']
idx_remove = [i for i, tr in enumerate(st)
              if tr.stats.station + '..' + tr.stats.channel in tr_remove]
if idx_remove:
    for idx in sorted(idx_remove, reverse=True):
        st.pop(idx)
        del pairs[idx]

# Load LFE data on Tim's catalog
templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
templates=templates[(templates['residual']<0.5)]
templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
templates = templates[(templates['OT'] >= startdate)
                    & (templates['OT'] < enddate)
                    & (templates['residual'] < 0.1)]
templates = templates.drop(columns=['dates', 'residual', 'dt'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
# To choose which templates
templates = templates.sort_values(by='N', ascending=False)
templates=templates.iloc[0:0+1]#[::3]
print(templates)

# Which stations detected the events
A = np.load(base_dir +'all_T_0.1_3.npy',allow_pickle=True).item()

# Collect information
num_detections = 1

# Generate the output files paths
info_file_path = os.path.join(base_dir, 'plots', f"{folder}", "info.txt")
output_file_path = os.path.join(base_dir, 'plots', f"{folder}", 'output.txt')

# If you want to reuse the detections as new templates and go through the process again
reuse_events=True
num_repeats=2

# Iterate over all templates
for idx, template_stats in templates.iterrows():
    # Initialization
    template=[]
    all_template=[]
    templ_idx=idx
    name = f'templ{idx}'
    xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
    # First template is made with the stations that detected the events
    # st2, pairs2 = crosscorr_tools.process_streams(st, template_stats, A)
    
    # Iterate over defined above stations
    for tr in st:
    # for tr in st2:
        # Template data
        start_templ = UTCDateTime(template_stats['OT']) + timedelta(seconds=8)
        end_templ = start_templ + timedelta(seconds=win_size)
        # Extract template data for each station
        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
        all_template.append(template.data)
        # Cross-correlate template with station data
        xcorr_template = autocorr_tools.correlate_template(
            tr.data, template.data,
            mode='valid', normalize='full', demean=True, method='auto'
        )
        # Ensure equal length for cross-correlation arrays
        if len(xcorr_template)<len(xcorr_full):
            xcorr_full=xcorr_full[:len(xcorr_template)]
        elif len(xcorr_template)>len(xcorr_full):
            xcorr_template=xcorr_template[:len(xcorr_full)]
        xcorr_full+=xcorr_template
        
        # Check if there are any NaN values
        if np.isnan(xcorr_full).any():
            # The NaN values are coming from the normalization done in
            # the cross-correlation function from obspy. 
            print(f"Trace {tr.id} contains NaN values.")

    # Network cross-correlation
    xcorrmean=xcorr_full/len(st)
    # xcorrmean=xcorr_full/len(st2)

    # If it goes over the next day, template not defined and end of the run
    if not all_template:
        break

    # Plot template time window on each station-channel combination
    template_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                              folder, name, 'template1', lastday)
    crosscorr_tools.plot_template(st, all_template, pairs, templ_idx, template_plot_filename)
    # crosscorr_tools.plot_template(st2, all_template, pairs2, templ_idx, template_plot_filename)

    # Find indices where the cross-correlation values are above the threshold
    mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
    thresh = 8 * mad

    # Determine if there are new detections
    windowlen = template.stats.npts / 2
    peaks, properties = find_peaks(xcorrmean, height=thresh, distance=windowlen)#, prominence=0.40)
    newdect = peaks
'''
    # If new detections
    if newdect.size > 1:
        cpt=1

        # Plot cross-correlation function
        crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                                  folder, name, 'crosscorr1', lastday)
        crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                        templ_idx, crosscorr_plot_filename, cpt)

        # Plot stacked traces
        stack_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                              folder, name, 'stack1', lastday)
        crosscorr_tools.plot_stacks(st, newdect, pairs, 
                                    templ_idx, stack_plot_filename, cpt)

        ## Writing in output.txt
        # Create UTCDateTime objects from the newevent values
        newevent = newdect*dt
        utc_times = [startdate + timedelta(seconds=event) for event in newevent]
        num_detections+=newevent.size
        # Save the cross-correlation values for each newevent
        cc_values = xcorrmean[newdect]
        #  Write detected events to output file
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            if os.stat(output_file_path).st_size == 0:
                output_file.write("starttime,templ,cc value,run\n")
            for i, utc_time in enumerate(utc_times):
                output_file.write(
                    f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                    f"{templ_idx},{cc_values[i]:.4f},1\n"
                )

        # Reuse detected events as templates by stacking
        print(f"Got {len(utc_times)} detections so let's reuse them as templates by stacking them!")
        for _ in range(num_repeats):
            cpt+=1
            print("Number of the next iteration :", cpt)
            # Plot new templates, cross-correlation, and new detections
            if reuse_events:
                xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
                stacked_templ = np.zeros((len(st),int(win_size*sampling_rate+1)))
                # Create stacked templates (1 template = stacked new detections)
                for idx, tr in enumerate(st):
                    # Template data
                    for utc_time in utc_times:
                        start_templ=UTCDateTime(utc_time)
                        end_templ = start_templ + timedelta(seconds=win_size)
                        # Extract template data for each station
                        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
                        # Normalize the template waveform
                        max_amplitude = np.max(np.abs(template.data))
                        if max_amplitude != 0:
                            template.data /= max_amplitude
                        # Add the normalized template to the stacked trace
                        stacked_templ[idx, :] += template.data

                    # Cross-correlate stacked template with station data
                    xcorr_template = autocorr_tools.correlate_template(
                        tr.data, stacked_templ[idx,:],
                        mode='valid', normalize='full', demean=True, method='auto'
                    )
                    # Ensure equal length for cross-correlation arrays
                    if len(xcorr_template)<len(xcorr_full):
                        xcorr_full=xcorr_full[:len(xcorr_template)]
                    elif len(xcorr_template)>len(xcorr_full):
                        xcorr_template=xcorr_template[:len(xcorr_full)]
                    xcorr_full+=xcorr_template

                # Calculate mean cross-correlation
                xcorrmean=xcorr_full/len(st)

                # Find indices where the cross-correlation values are above the threshold
                mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
                thresh = 8 * mad

                # Determine if there are new detections
                peaks, properties = find_peaks(xcorrmean, height=thresh, distance=windowlen)#, prominence=0.40)
                newdect = peaks

                # If new detections
                if newdect.size > 1:
                    # Plot cross-correlation function for new detections
                    crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'crosscorr{cpt}', lastday)
                    crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                                    templ_idx, crosscorr_plot_filename, cpt)

                    # Plot stacked traces for new detections
                    stack_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'stack{cpt}', lastday)
                    crosscorr_tools.plot_stacks(st, newdect, pairs,
                                                templ_idx, stack_plot_filename, cpt=cpt)

                    ## Writing in output.txt
                    # Create UTCDateTime objects from the newevent values
                    newevent = newdect*dt
                    utc_times = [startdate + timedelta(seconds=event) for event in newevent]
                    print(f"Got {len(utc_times)} new detections with the new templates!")
                    # Save the cross-correlation values for each newevent
                    cc_values = xcorrmean[newdect]
                    #  Write the newevent and additional columns to the output file
                    with open(output_file_path, "a", encoding="utf-8") as output_file:
                        for i, utc_time in enumerate(utc_times):
                            output_file.write(
                                f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                                f"{templ_idx},{cc_values[i]:.4f},1\n"
                            )

    # Follow the advancement
    print(f"Processed template {templ_idx + 1}/{len(templates)}")

# Calculate and print script execution time
end_script = time.time()
script_execution_time = end_script - startscript
print(f"Script execution time: {script_execution_time:.2f} seconds")

# Write the info of the run in the output file
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {UTCDateTime(st[0].stats.starttime)}"
               f"- {UTCDateTime(st[0].stats.endtime)}\n\n")
    file.write(f"Stations and Channels Used: {pairs}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write(f"Total of detections: {num_detections} (with redundant times)\n")
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')
'''