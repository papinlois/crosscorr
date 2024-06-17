#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

Version for the cluster on talapas of crosscorr.py.
    
As of 17/06/24.
"""

# ================ Initialization ================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from scipy.signal import find_peaks
import autocorr_tools
import crosscorr_tools

startscript = time.time()

# Define the base directory
base_dir = "/home/lpapin/crosscorr"
folder = "SSE_2005"
diff = "aug_PO" # If different network tested
which = 'talapas'
# Generate the output files paths
info_file_path = os.path.join(base_dir, 'plots', f"{folder}", f'{diff}',f"info_{diff}.txt")
output_file_path = os.path.join(base_dir, 'plots', f"{folder}", f'{diff}',  f"output_{diff}.txt")

# Define the network
from network_configurations_talapas import network_config
stas = [station for value in network_config.values() for station in value['stations']]

# ================ Events ================

startall = datetime.strptime("20050903", "%Y%m%d")
endall = datetime.strptime("20050925", "%Y%m%d")
# Load LFE data on Tim's catalog
templates=pd.read_csv('./EQloc_001_0.1_3_S.csv', index_col=0)
templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
templates = templates[(templates['OT'] >= startall)
                    & (templates['OT'] < endall)
                    & (templates['residual'] < 0.1)]
templates = templates.drop(columns=['residual', 'dt'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
# 5 random templates per day with highest N values
random_templates = templates.groupby(templates['OT'].dt.date).apply(
    crosscorr_tools.select_random_templates)
templates = random_templates.groupby(random_templates['OT'].dt.date).apply(
    lambda x: x.nlargest(20, 'N'))
templates.index = templates.index.droplevel(level=[0, 1])
templates.sort_index(ascending=True)
print(templates)
# Rearranged so 1 per day at the time on the full period
grouped = templates.groupby(templates['OT'].dt.date)
max_templates_per_day = grouped.size().max()
rearranged_templates = []
for i in range(max_templates_per_day):
    for name, group in grouped:
        if i < len(group):
            rearranged_templates.append(group.iloc[i])
templates = pd.concat(rearranged_templates, axis=1).T
templates.reset_index(drop=True, inplace=True)
print(templates)

# ================ Template-matching parameters ================

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 10
# Days around the template
days_crosscorr = 3
# How may times do you iterate?
reuse_events=False
num_repeats = 3
# Start time definition for the windows
windows=crosscorr_tools.create_window(templates, stas, base_dir, diff)
print(windows)
interval = 'P' #P for min_p_wave+win_size or S for percentile_75th_s_wave-win_size
# Track results
cpttempl = 0
failed_templ = []
total_events=0
templates_dict={}

# ================ Template-matching process ================

for idx, template_stats in templates.iterrows():
    ## Time and data parameters
    time_event = UTCDateTime(template_stats['OT'])
    # Adjust data window to include cross_days days before and after the template's starttime
    start_data = (time_event - timedelta(days=days_crosscorr)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    end_data = (time_event + timedelta(days=days_crosscorr+1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    # Get the streams and preprocess
    dates_of_interest = []
    current_date = start_data
    while current_date <= end_data:
        dates_of_interest.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    st, pairs = crosscorr_tools.get_traces(
        network_config, dates_of_interest, base_dir, which=which)
    # Remove the bad data: has to be specfic to the period you're looking at
    tr_remove = ['PGC..BHE','SHVB..HHE','SHVB..HHN','SNB..BHE',
                   'TWKB..HHE','VGZ..BHE','YOUB..HHZ']
    st, pairs = crosscorr_tools.remove_stations(st, pairs, tr_remove)
    st = crosscorr_tools.process_data(
        st, start_data, end_data, sampling_rate, freqmin, freqmax)
    print(st.__str__(extended=True))

    ## Initialization of template-matching parameters
    template=[]
    all_template=[]
    templ_idx=idx
    name = f'templ{idx}'
    N = template_stats['N']
    day = time_event.strftime("%Y%m%d")
    xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
    mask=np.zeros(len(xcorr_full))
    ### Can do a function for which interval is the best?
    if interval=='P':
        offset = windows.loc[templ_idx, 'timedelta_P']
        start_templ = time_event + timedelta(seconds=offset)
        end_templ = start_templ + timedelta(seconds=win_size)
        count = windows.loc[templ_idx, 'nb_stas_P']
        print(f"{count} stations with P and S in the interval P.")
    elif interval=='S':
        offset = windows.loc[templ_idx, 'timedelta_S']
        end_templ = time_event + timedelta(seconds=offset)
        start_templ = end_templ - timedelta(seconds=win_size)
        count = windows.loc[templ_idx, 'nb_stas_S']
        print(f"{count} stations with P and S in the interval S.")

    # Update templates_dict with the information for the current template
    templates_dict[idx] = {
        'OT': str(template_stats['OT']),
        'lon': template_stats['lon'],
        'lat': template_stats['lat'],
        'depth': template_stats['depth'],
        'N': int(N),
        'offset': int(offset),
        'count': int(count)
    }

    ## Cross-correlation process
    for tr in st:
        # Extract trace data for the cross-correlation
        data_window = tr.copy().trim(starttime=start_data, endtime=end_data)
        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
        all_template.append(template.data)
        # Cross-correlate template with station data
        xcorr_template = autocorr_tools.correlate_template(
            data_window.data, template.data,
            mode='valid', normalize='full', demean=True, method='auto'
        )
        # Ensure equal length for cross-correlation arrays
        xcorr_full, xcorr_template, mask = crosscorr_tools.check_length(
            xcorr_full, xcorr_template, mask)
        # Check if there are any NaN values and make it 0
        xcorr_template, mask = crosscorr_tools.check_xcorr_template(
            xcorr_template, mask)
        xcorr_full+=xcorr_template

    # Check if there is a lack of channels enough to remove the cc values
    xcorr_full, mask = crosscorr_tools.check_xcorr_full(
        xcorr_full, mask)

    # Network cross-correlation
    xcorrmean=xcorr_full/mask

    # Find indices where the cross-correlation values are above the threshold
    mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
    thresh = 8 * mad

    # Determine if there are new detections
    windowlen = template.stats.npts / 2 ###
    newdect, _ = find_peaks(xcorrmean, height=thresh, distance=windowlen)

    ## If new detections
    if 2 <= newdect.size <= 1000:
        cpt=1 # Iteration number

        # Plot template time window on each station-channel combination
        template_plot_filename = crosscorr_tools.build_file_path(
            base_dir, folder, diff, name, 'template1', day)
        crosscorr_tools.plot_template(
            all_template, pairs, time_event, N, sampling_rate,
            templ_idx, template_plot_filename)

        # Plot cross-correlation function
        crosscorr_plot_filename = crosscorr_tools.build_file_path(
            base_dir, folder, diff, name, 'crosscorr1', day)
        crosscorr_tools.plot_crosscorr(
            xcorrmean, thresh, dt, newdect, templ_idx,
            crosscorr_plot_filename, cpt=cpt, mask=mask)

        # Plot stacked traces
        stack_plot_filename = crosscorr_tools.build_file_path(
            base_dir, folder, diff, name, 'stack1', day)
        crosscorr_tools.plot_stacks(
            st, newdect, pairs, templ_idx, stack_plot_filename, cpt)

        # Create UTCDateTime objects from the newevent values
        newevent = newdect*dt
        utc_times = [start_data + timedelta(seconds=event) for event in newevent]

        ## Reuse detected events as templates by stacking
        print(f"Got {len(utc_times)} detections so let's reuse them as templates by stacking them!")
        if reuse_events:
            # Plot new templates, cross-correlation, and new detections
            for _ in range(num_repeats):
                cpt+=1
                print("Number of the next iteration :", cpt)
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
                    xcorr_template, mask = crosscorr_tools.check_xcorr_template(
                        xcorr_template, mask)
                    xcorr_full+=xcorr_template

                # Calculate mean cross-correlation
                xcorrmean=xcorr_full/mask

                # Find indices where the cross-correlation values are above the threshold
                mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
                thresh = 8 * mad

                # Determine if there are new detections
                newdect, _ = find_peaks(xcorrmean, height=thresh, distance=windowlen)

                ## If new detections ###
                if 2 <= newdect.size <= 1000:
                    # Plot cross-correlation function for new detections
                    crosscorr_plot_filename = crosscorr_tools.build_file_path(
                        base_dir, folder, diff, name, f'crosscorr{cpt}', day)
                    crosscorr_tools.plot_crosscorr(
                        xcorrmean, thresh, dt, newdect, templ_idx,
                        crosscorr_plot_filename, cpt=cpt, mask=mask)

                    # Plot stacked traces for new detections
                    stack_plot_filename = crosscorr_tools.build_file_path(
                        base_dir, folder, diff, name, f'stack{cpt}', day)
                    crosscorr_tools.plot_stacks(
                        st, newdect, pairs, templ_idx, stack_plot_filename, cpt=cpt)

                    # Create UTCDateTime objects from the newevent values
                    newevent = newdect*dt
                    utc_times = [start_data + timedelta(seconds=event) for event in newevent]
                    print(f"Got {len(utc_times)} new detections with the new templates!")
                    num_events=len(utc_times)

                else:
                    # 0 detections or too much
                    failed_templ.append(str(templ_idx))

        ## Writing in output.txt for the last iteration
        num_events=len(utc_times)
        print(f"Total of {num_events} new detections for template {templ_idx}.")
        total_events+=num_events
        # Save the cross-correlation values for each newevent
        cc_values = xcorrmean[newdect]
        #  Write the newevent and additional columns to the output file
        with open(output_file_path, "a", encoding="utf-8") as output_file:
            if os.stat(output_file_path).st_size == 0:
                output_file.write("starttime,template,coeff,run\n")
            for i, utc_time in enumerate(utc_times):
                output_file.write(
                    f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                    f"{templ_idx},{cc_values[i]:.4f},{cpt}\n"
                )
    else:
        # 0 detections or too much
        failed_templ.append(str(templ_idx))
        cpt=0

    # Follow the advancement
    cpttempl+=1
    print(f"Template {templ_idx} processed ({cpttempl}/{len(templates)})")

# ================ All outputs ================

script_execution_time = time.time() - startscript
# Write the info of the run in the output file info.txt
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {startall.strftime('%Y-%m-%d')}"
               f"-{endall.strftime('%Y-%m-%d')} with "
               f"{days_crosscorr} days around the event\n\n")
    file.write(f"Stations : {stas}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write(f"Script execution time: {script_execution_time:.2f} seconds\n\n")
    if failed_templ:
        file.write(f"Templates that didn't satisfy the requirement "
                   f"of detections (2<=new detections<1000) ({len(failed_templ)}/"
                   f"{len(templates)}): {', '.join(failed_templ)}\n")
    file.write(f"Number total of detections: {total_events} \n\n")
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')

# Save the templates_dict to a JSON file
with open('templates_dict.json', 'w') as json_file:
    json.dump(templates_dict, json_file, indent=4)
