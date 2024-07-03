#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

This script performs a network cross-correlation analysis on seismic data with
already known events as templates, in order to detect new events.
Functions are from autocorrelation and cross-correlation tools modules.

To be used:
    - Change the paths, the stations, the period, the catalog/templates,
      all parameters for the processing of the streams and for time windows
    
As of 16/05/24.
"""

import os
from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from obspy import UTCDateTime
from scipy.signal import find_peaks
import autocorr_tools
import crosscorr_tools

# import matplotlib
# matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
folder = "bostock" ###
# Generate the output files paths
info_file_path = os.path.join(base_dir, 'plots', f"{folder}", "info.txt")
output_file_path = os.path.join(base_dir, 'plots', f"{folder}", 'output.txt')

# Define the network
from network_configurations import network_config

# Days of data
startdate = datetime.strptime("20050903", "%Y%m%d")
enddate = datetime.strptime("20050925", "%Y%m%d")
days_crosscorr = 3 # +/- around the template
dates_of_interest = []
current_date = startdate
while current_date <= enddate:
    dates_of_interest.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)
lastday = dates_of_interest[-1] #For filenames
# To assure same index as the one in the arrival times file
startall = datetime.strptime("20050101", "%Y%m%d")
endall = datetime.strptime("20051231", "%Y%m%d")

# Get the streams
st = crosscorr_tools.get_traces(network_config, dates_of_interest, base_dir)

# List of stations/channels to analyze
pairs=[]
stas=[]
for tr in st:
    pairs.append(tr.id[3:])
    stas.append(tr.stats.station)
stas=list(set(stas))

# Remove the bad data: has to be specfic to the period you're looking at ###
tr_remove = ['PGC..BHE','SHVB..HHE','SHVB..HHN','SNB..BHE',
             'TWKB..HHE','VGZ..BHE','YOUB..HHZ']
st, pairs = crosscorr_tools.remove_stations(st, pairs, tr_remove)

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 10

# # Plot the streams to see how the data looks like; help to understand the quality
# # of the results and to remove bad data that are difficult to automatically
# # remove with parameters in functions. NB: memory expensive (a lot!)
# data_plot_filename = os.path.join(base_dir,f'plots/{folder}/data_plot.png')
# crosscorr_tools.plot_data(st, pairs, data_plot_filename)

# Preprocess the data
st = crosscorr_tools.process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax)
print(st.__str__(extended=True))

# Load LFE data on Bostock's catalog
templates = pd.read_csv('lfe_svi.txt', index_col=0, dtype={'date': str, 'hour': str, 'lfe_family' : str})
templates['date'] = '20' + templates['date']
templates['date'] = pd.to_datetime(templates['date'], format='%Y%m%d')
templates['hour'] = templates['hour'].str.zfill(2)
templates['OT'] = templates['date'] + pd.to_timedelta(templates['hour'].astype(int), unit='h') + pd.to_timedelta(templates['second'], unit='s')
templates = templates[(templates['OT'] >= startdate) & (templates['OT'] < enddate)]
templates = templates.drop(columns=['Mw','hour','second','date'])
templates = templates.sort_values(by='OT', ascending=True)
templates.reset_index(inplace=True)
templates.index.name = 'Index'
print(templates)

# Get the parameters for the window of each template
# !! Check the arrival times used in function before run
windows=crosscorr_tools.create_window(templates, stas, base_dir)
print(windows)

# If you want to reuse the detections as new templates and
# go through the process again, how many times?
reuse_events=True
num_repeats=3
interval = 'S' #P for min_p_wave+win_size or S for percentile_75th_s_wave-win_size

# Iterate over all templates
for idx, template_stats in templates.iterrows():
    # Initialization
    template=[]
    all_template=[]
    templ_idx=idx
    name = f'templ{idx}'
    xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
    mask=np.zeros(len(xcorr_full))
    if interval=='P':
        offset = windows.loc[templ_idx, 'timedelta_P']
    elif interval=='S':
        offset = windows.loc[templ_idx, 'timedelta_S']
    time_event = UTCDateTime(template_stats['OT'])
    # Iterate over all stations and channels combination
    for tr in st:
        if interval=='P':
            start_templ = time_event + timedelta(seconds=offset)
            end_templ = start_templ + timedelta(seconds=win_size)
        elif interval=='S':
            end_templ = time_event + timedelta(seconds=offset)
            start_templ = end_templ - timedelta(seconds=win_size)
        # Extract template data for each station
        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
        all_template.append(template.data)
        # # Adjust data window to include cross_days days before and after the template's starttime
        start_data = (start_templ - timedelta(days=days_crosscorr)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        end_data = (end_templ + timedelta(days=days_crosscorr+1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        data_window = tr.copy().trim(starttime=start_data, endtime=end_data)
        # Cross-correlate template with station data
        xcorr_template = autocorr_tools.correlate_template(
            data_window.data, template.data,
            mode='valid', normalize='full', demean=True, method='auto'
        )
        # Ensure equal length for cross-correlation arrays
        xcorr_full, xcorr_template, mask = crosscorr_tools.check_length(
            xcorr_full, xcorr_template, mask)
        # Check if there are any NaN values and make it 0
        xcorr_template, mask = crosscorr_tools.check_xcorr(
            xcorr_template, mask)
        xcorr_full+=xcorr_template

    # Network cross-correlation
    xcorrmean=xcorr_full/mask

    # Plot template time window on each station-channel combination
    template_plot_filename = crosscorr_tools.build_file_path(
        base_dir, folder, name, 'template1', lastday)
    crosscorr_tools.plot_template(
        all_template, pairs, time_event, sampling_rate,
        templ_idx, template_plot_filename)

    # Find indices where the cross-correlation values are above the threshold
    mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
    thresh = 8 * mad

    # Determine if there are new detections
    windowlen = template.stats.npts / 2 ###
    newdect, _ = find_peaks(xcorrmean, height=thresh, distance=windowlen)

    # Plot cross-correlation function
    crosscorr_plot_filename = crosscorr_tools.build_file_path(
        base_dir, folder, name, 'crosscorr1', lastday)
    crosscorr_tools.plot_crosscorr(
        xcorrmean, thresh, dt, newdect, templ_idx,
        crosscorr_plot_filename, cpt=1, mask=mask)

    # If new detections
    if newdect.size > 1:
        cpt=1 # Iteration number

        # Plot stacked traces
        stack_plot_filename = crosscorr_tools.build_file_path(
            base_dir, folder, name, 'stack1', lastday)
        crosscorr_tools.plot_stacks(
            st, newdect, pairs, templ_idx, stack_plot_filename, cpt)

        # Create UTCDateTime objects from the newevent values
        newevent = newdect*dt
        utc_times = [startdate + timedelta(seconds=event) for event in newevent]

        # Reuse detected events as templates by stacking
        print(f"Got {len(utc_times)} detections so let's reuse them as templates by stacking them!")
        for _ in range(num_repeats):
            cpt+=1
            print("Number of the next iteration :", cpt)
            # Plot new templates, cross-correlation, and new detections
            if reuse_events:
                # Initialization
                xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
                stacked_templ = np.zeros((len(st),int(win_size*sampling_rate+1)))
                mask=np.zeros(len(xcorr_full))
                # Create stacked templates (1 template = stacked new detections)
                for idxx, tr in enumerate(st):
                    # Template data
                    for _, utc_time in enumerate(utc_times):
                        start_templ = UTCDateTime(utc_time)
                        end_templ = start_templ + timedelta(seconds=win_size)
                        # Extract template data for each station
                        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
                        # Normalize the template waveform
                        max_amplitude = np.max(np.abs(template.data))
                        if max_amplitude != 0:
                            template.data /= max_amplitude
                        stacked_templ[idxx, :] += template.data

                    data_window = tr.copy().trim(starttime=start_data, endtime=end_data)
                    # Cross-correlate stacked template with station data
                    xcorr_template = autocorr_tools.correlate_template(
                        data_window.data, stacked_templ[idxx,:],
                        mode='valid', normalize='full', demean=True, method='auto'
                    )
                    # Ensure equal length for cross-correlation arrays
                    xcorr_full, xcorr_template, mask = crosscorr_tools.check_length(
                        xcorr_full, xcorr_template, mask)
                    # Check if there are any NaN values and make it 0
                    xcorr_template, mask = crosscorr_tools.check_xcorr(
                        xcorr_template,mask)
                    xcorr_full+=xcorr_template

                # Calculate mean cross-correlation
                xcorrmean=xcorr_full/mask

                # Find indices where the cross-correlation values are above the threshold
                mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
                thresh = 8 * mad

                # Determine if there are new detections
                newdect, _ = find_peaks(xcorrmean, height=thresh, distance=windowlen)

                # If new detections
                if newdect.size > 1:
                    # Plot cross-correlation function for new detections
                    crosscorr_plot_filename = crosscorr_tools.build_file_path(
                        base_dir, folder, name, f'crosscorr{cpt}', lastday)
                    crosscorr_tools.plot_crosscorr(
                        xcorrmean, thresh, dt, newdect, templ_idx,
                        crosscorr_plot_filename, cpt=cpt, mask=mask)

                    # Plot stacked traces for new detections
                    stack_plot_filename = crosscorr_tools.build_file_path(
                        base_dir, folder, name, f'stack{cpt}', lastday)
                    crosscorr_tools.plot_stacks(
                        st, newdect, pairs, templ_idx, stack_plot_filename, cpt=cpt)

                    # Create UTCDateTime objects from the newevent values
                    newevent = newdect*dt
                    utc_times = [startdate + timedelta(seconds=event) for event in newevent]
                    print(f"Got {len(utc_times)} new detections with the new templates!")
                    num_events=len(utc_times)

        ## Writing in output.txt for the last iteration
        # Create UTCDateTime objects from the newevent values
        newevent = newdect*dt
        utc_times = [startdate + timedelta(seconds=event) for event in newevent]
        num_events=len(utc_times)
        print(f"Total of {num_events} new detections for template {templ_idx}.")
        # Save the cross-correlation values for each newevent
        cc_values = xcorrmean[newdect]
        #  Write the newevent and additional columns to the output file
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            if os.stat(output_file_path).st_size == 0:
                output_file.write("starttime,template,cc value,run\n")
            for i, utc_time in enumerate(utc_times):
                output_file.write(
                    f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                    f"{templ_idx},{cc_values[i]:.4f},{cpt}\n"
                )
    else:
        cpt=0 # No detections

    # Follow the advancement
    print(f"Template {templ_idx} processed "
          f"({templ_idx-templates.iloc[-1].name+len(templates)}/{len(templates)})" )

# Write the info of the run in the output file info.txt
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {UTCDateTime(st[0].stats.starttime)}"
               f"- {UTCDateTime(st[0].stats.endtime)}\n\n")
    file.write(f"Stations and Channels Used: {pairs}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')
