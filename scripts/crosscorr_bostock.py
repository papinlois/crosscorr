 # -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

This script performs a network cross-correlation analysis on seismic data with
already known events as templates, in order to detect new events.
Functions are from autocorrelation and cross-correlation tools modules.

As of 02/18/23.

"""

import os
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
folder = "bostock2"
# Generate the output files paths
info_file_path = os.path.join(base_dir, 'plots', f"{folder}", "info.txt")
output_file_path = os.path.join(base_dir, 'plots', f"{folder}", 'output.txt')

# Define the network : C8 + PO
from network_configurations_test import network_config

# Days of data
startdate = datetime.strptime("20050903", "%Y%m%d")
enddate = datetime.strptime("20050908", "%Y%m%d")
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
win_size = 8

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax)
print(st.__str__(extended=True))

# List of stations/channels to analyze
pairs=[]
for tr in st:
    iid = tr.id
    identifier = iid[3:]
    pairs.append(identifier)

# Remove the bad data
tr_remove = ['PGC..BHE', 'SHVB..HHE', 'SHVB..HHN','SNB..BHE','TWKB..HHE','VGZ..BHE','YOUB..HHZ']
idx_remove = [i for i, tr in enumerate(st)
              if tr.stats.station + '..' + tr.stats.channel in tr_remove]
if idx_remove:
    for idx in sorted(idx_remove, reverse=True):
        st.pop(idx)
        del pairs[idx]

# # Load LFE data on Tim's catalog
# templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
# templates=templates[(templates['residual']<0.5)]
# templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
# templates = templates[(templates['OT'] >= startdate)
#                     & (templates['OT'] < enddate)
#                     & (templates['residual'] < 0.1)]
# templates = templates.drop(columns=['dates', 'residual', 'dt'])
# templates.reset_index(inplace=True, drop=True)
# templates.index.name = 'Index'
# # To choose which templates
# templates = templates.sort_values(by='N', ascending=False)
# templates=templates.iloc[0:5+1]
# print(templates)

# Load LFE data on Bostock's catalog
templates = pd.read_csv('lfe_svi.txt', index_col=0, dtype={'date': str, 'hour': str, 'lfe_family':str})
templates['date'] = '20' + templates['date']
templates['date'] = pd.to_datetime(templates['date'], format='%Y%m%d')
templates['hour'] = templates['hour'].str.zfill(2)
templates['OT'] = templates['date'] + pd.to_timedelta(templates['hour'].astype(int), unit='h') + pd.to_timedelta(templates['second'], unit='s')
templates = templates[(templates['OT'] >= startdate) & (templates['OT'] < enddate)]
templates = templates.drop(columns=['Mw','hour','second','date'])
templates = templates.sort_values(by='OT', ascending=True)
# templates = templates.sort_values(by='lfe_family', ascending=False)
# templates = templates.sort_values(by=['lfe_family', 'OT'], ascending=[True, True])
templates.reset_index(inplace=True)
templates.index.name = 'Index'
## To choose which templates
# # 1 event per family
templates=templates.groupby('lfe_family').first().reset_index()
# # 1 template
# templates=templates.iloc[2655:2657+1]
# templates=templates[1:1+1] # pour OT descending #1256
# # 1 template every 50
# templates=templates[::5]
# # 1 template, 1 event per day
# templates = templates[templates['lfe_family'] == '001']
# templates = templates.groupby(templates['OT'].dt.date).first()
# templates['Index'] = range(len(templates))
# templates.set_index('Index', inplace=True)
# templates=templates.iloc[0:0+1]
# templates=templates[::1000]
templates=templates[:40]
print(templates)

# If you want to reuse the detections as new templates and g
# go through the process again, how many times?
reuse_events=True
num_repeats=2

# Iterate over all templates
for idx, template_stats in templates.iterrows():
    # Initialization
    template=[]
    all_template=[]
    templ_idx=idx
    family_idx=templates['lfe_family'].iloc[0]
    name = f'templ{idx}'
    xcorr_full=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
    mask=np.zeros(len(xcorr_full))

    # Iterate over all stations and channels combination
    for tr in st:
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
        # Ensure equal length for cross-correlation arrays ###
        xcorr_full, xcorr_template, mask = crosscorr_tools.check_length(xcorr_full, xcorr_template, mask)
        # Check if there are any NaN values and make it 0
        xcorr_template, mask = crosscorr_tools.check_xcorr(xcorr_template, mask)
        xcorr_full+=xcorr_template

    # Network cross-correlation
    xcorrmean=xcorr_full/mask

    # Plot template time window on each station-channel combination
    template_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                              folder, name, 'template1', lastday)
    crosscorr_tools.plot_template(st, all_template, pairs, templ_idx, template_plot_filename)

    # Find indices where the cross-correlation values are above the threshold
    mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
    thresh = 8 * mad

    # Determine if there are new detections
    windowlen = template.stats.npts / 2
    peaks, properties = find_peaks(xcorrmean, height=thresh, distance=windowlen)#, prominence=0.40)
    newdect = peaks
    
    # cpt=0
    # If new detections
    if 1 < newdect.size < 250:
    # if newdect.size > 1:
        cpt=1

        # Plot cross-correlation function
        crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                                  folder, name, 'crosscorr1', lastday)
        crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                        templ_idx, crosscorr_plot_filename, cpt, mask=mask)

        # Plot stacked traces
        stack_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                              folder, name, 'stack1', lastday)
        crosscorr_tools.plot_stacks(st, newdect, pairs,
                                    templ_idx, stack_plot_filename, cpt)

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
                for idx, tr in enumerate(st):
                    # Template data
                    for _, utc_time in enumerate(utc_times):
                        temp_template=[]
                        start_templ=UTCDateTime(utc_time)
                        end_templ = start_templ + timedelta(seconds=win_size)
                        # Extract template data for each station
                        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
                        # temp_template.append(template.data)
                        # Normalize the template waveform
                        max_amplitude = np.max(np.abs(template.data))
                        if max_amplitude != 0:
                            template.data /= max_amplitude
                        # Add the normalized template to the stacked trace if not main frequency between 1-2Hz
                        # condition_met = crosscorr_tools.check_peak_frequency(temp_template)
                        # if condition_met is True:
                        stacked_templ[idx, :] += template.data

                    # Cross-correlate stacked template with station data
                    xcorr_template = autocorr_tools.correlate_template(
                        tr.data, stacked_templ[idx,:],
                        mode='valid', normalize='full', demean=True, method='auto'
                    )
                    # Ensure equal length for cross-correlation arrays
                    xcorr_full, xcorr_template, mask = crosscorr_tools.check_length(xcorr_full, xcorr_template, mask)
                    # Check if there are any NaN values and make it 0
                    xcorr_template, mask = crosscorr_tools.check_xcorr(xcorr_template,mask)
                    xcorr_full+=xcorr_template

                # Calculate mean cross-correlation
                xcorrmean=xcorr_full/mask

                # Find indices where the cross-correlation values are above the threshold
                mad = np.nanmedian(np.abs(xcorrmean - np.nanmedian(xcorrmean)))
                thresh = 8 * mad

                # Determine if there are new detections
                peaks, properties = find_peaks(xcorrmean, height=thresh, distance=windowlen)#, prominence=0.40)
                newdect = peaks

                # If new detections
                # if newdect.size > 1:
                if 1 < newdect.size < 250:
                    # Plot cross-correlation function for new detections
                    crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'crosscorr{cpt}', lastday)
                    crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                                    templ_idx, crosscorr_plot_filename, cpt=cpt, mask=mask)

                    # Plot stacked traces for new detections
                    stack_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'stack{cpt}', lastday)
                    crosscorr_tools.plot_stacks(st, newdect, pairs,
                                                templ_idx, stack_plot_filename, cpt=cpt)

                    # Create UTCDateTime objects from the newevent values
                    newevent = newdect*dt
                    utc_times = [startdate + timedelta(seconds=event) for event in newevent]
                    print(f"Got {len(utc_times)} new detections with the new templates!")
                    num_events=len(utc_times)
                # else:
                #     break # Too much detections we want to avoid useless computation
                #     # When it's hapenning it can be due to the data itself specially
                #     # now because we are removing the low-frequencies
                #     print(f"Computation stopped for template {templ_idx}.")
    # else:
    #     continue
    #     print(f"Computation stopped for template {templ_idx}.")

    ## Writing in output.txt for the last iteration
    # Create UTCDateTime objects from the newevent values
    newevent = newdect*dt
    utc_times = [startdate + timedelta(seconds=event) for event in newevent]
    num_events=len(utc_times)
    print(f"Total of {num_events} new detections for template {templ_idx}.")
    # Save the cross-correlation values for each newevent
    cc_values = xcorrmean[newdect]
    #  Write the newevent and additional columns to the output file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for i, utc_time in enumerate(utc_times):
            output_file.write(
                f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                f"{family_idx},{templ_idx},{cc_values[i]:.4f},{cpt}\n"
            )

    # Follow the advancement
    print(f"Processed template {templ_idx + 1}/{len(templates)}")

# Write the info of the run in the output file info.txt
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {UTCDateTime(st[0].stats.starttime)}"
               f"- {UTCDateTime(st[0].stats.endtime)}\n\n")
    file.write(f"Stations and Channels Used: {pairs}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write(f"Total of detections: {len(utc_times)}\n")
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')
