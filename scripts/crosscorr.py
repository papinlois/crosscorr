# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

This script performs a network cross-correlation analysis on seismic data with
already known events as templates, in order to detect new events.
Functions are from autocorrelation and cross-correlation tools modules.

Description:
1. Creates an info.txt file with relevant information about the data.
2. Loads seismic data from specified stations and channels for given dates.
3. Preprocesses the data by trimming, interpolating, detrending, and filtering.
4. Plots the full streams of seismic data.
5. Load earthquake catalog data and select specific templates.
6. Plots the station locations as well as the templates locations (optional).
7. Iterate over each template:
    8. Extract template data for each station and cross-correlate with station data.
    9. Plot the template time window on each station-channel combination.
    10. Determine correlations above a threshold, and identify new detections.
    11. If new detections exist, plot cross-correlation function, stacked traces,
    and output to 'output.txt'.
        12. Optionally, reuse detected events as templates by stacking them and repeat the process.
        13. If new detections with stacked templates exist, plot cross-correlation
        function, stacked traces, and output to 'output.txt'.
14. Output the script execution time, list of stations and channels used,
and additional information.

Note: This code is made for cross-correlation for several days of continuous
data. If you want a day, you still have to enter 2 dates of interests.

As of 12/15/23.

"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from obspy import UTCDateTime
import autocorr_tools
import crosscorr_tools

import matplotlib
matplotlib.use('Agg')

# Define the base directory
base_dir = "C:/Users/papin/Desktop/phd/"

# Plot station locations
locfile = pd.read_csv(os.path.join(base_dir, 'stations.csv'))
locs = locfile[['Name', 'Longitude', 'Latitude','Network']].values

# Start timer
startscript = time.time()

# Define the network configurations (CN & PB)
# NLLB removed, careful for some of the B stations (B010 + B926)
network_config = {
    'CN1': {
        'stations': ['LZB', 'PGC', 'SNB'],
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'CN2': {
        'stations': ['YOUB', 'PFB'],
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'PB': {
        'stations': ['B001', 'B009', 'B011'],
        'channels': ['EH1', 'EH2', 'EHZ'],
        'filename_pattern': '{station}.PB.{year}.{julian_day}'
    }
}

# Days of data
date_of_interests = ["20100516","20100517","20100518","20100519","20100520"]
startdate=datetime.strptime(date_of_interests[0], "%Y%m%d")
enddate=startdate+timedelta(days=len(date_of_interests) - 1)
lastday=date_of_interests[-1] #For filenames

# Frequency range, sampling_rate, and time window
freqmin = 2.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 30

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, sampling_rate, freqmin, freqmax, startdate, enddate)

# List of stations/channels to analyze
stas = [network_config[key]['stations'] for key in network_config]
channels = [network_config[key]['channels'] for key in network_config]
pairs = [f"{sta}..{cha}" for sta_list, cha_list in zip(stas, channels)
          for sta in sta_list for cha in cha_list]

# To create beforehand
folder = "test4" #f"{lastday}" #Last day of the series
# folder = f"{lastday} {len(date_of_interests)}days" #Last day of the series

# Plot all the streams and get all the combination of sta/cha
data_plot_filename = os.path.join(
    base_dir,
    f'plots/{folder}/data_plot_{lastday}.png'
)
# crosscorr_tools.plot_data(st, stas, channels, data_plot_filename)

# Load LFE data on Tim's catalog
templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
templates=templates[(templates['residual']<0.5)]
templates['datetime']=pd.to_datetime(templates['OT'])
templates = templates[(templates['datetime'] >= st[0].stats.starttime.datetime)
                    & (templates['datetime'] < st[0].stats.endtime.datetime)
                    & (templates['residual'] < 0.1)]
templates = templates.drop(columns=['dates', 'N', 'residual','starttime'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
# To choose which templates
templates=templates[120:120+1]#.iloc[::10]

# Plot locations of events and stations
events = templates[['lon', 'lat', 'depth', 'datetime']]
# crosscorr_tools.plot_locations(locs, base_dir, events=events)

# Collect information
info_lines = []  # Store lines of information
num_detections = 0

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

    # Iterate over all stations and channels combination
    for tr in st:
        # Template data
        start_templ = UTCDateTime(template_stats['datetime'] + timedelta(seconds=10))
        end_templ = start_templ + timedelta(seconds=win_size)
        if end_templ.day > enddate.day:
            print('Last template has an ending time on a wrong day: not processed.')
            break
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
        xcorr_full+=xcorr_template

    # Network cross-correlation
    xcorrmean=xcorr_full/len(st)

    # If it goes over the next day, template not defined and end of the run
    if not all_template:
        break

    # Plot template time window on each station-channel combination
    template_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                             folder, name, 'template1', lastday)
    crosscorr_tools.plot_template(st, all_template, pairs, templ_idx, template_plot_filename)

    # Find indices where the cross-correlation values are above the threshold
    mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
    thresh = 8 * mad
    aboves = np.where(xcorrmean > thresh)

    # Check if there are correlations above the threshold
    if aboves[0].size > 0:
        # Determine if there are new detections
        windowlen = template.stats.npts
        inds = aboves[0]
        clusters = autocorr_tools.clusterdects(inds, windowlen)
        newdect = autocorr_tools.culldects(inds, clusters, xcorrmean)
        # Find the index of the maximum value in newdect for plot reason
        max_index = np.argmax(xcorrmean[newdect])

        # Check if there are new detections
        if newdect.size > 1:
            # Plot cross-correlation function
            crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                                      folder, name, 'crosscorr1', lastday)
            crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                            max_index, name,
                                            lastday, crosscorr_plot_filename)

            # Plot stacked traces
            stack_plot_filename = crosscorr_tools.build_file_path(base_dir,
                                                                  folder, name, 'stack1', lastday)
            crosscorr_tools.plot_stacks(st, template, newdect, pairs,
                                        templ_idx, stack_plot_filename, cpt=1)

            ## Writing in output.txt
            # Create UTCDateTime objects from the newevent values
            newevent = newdect*dt
            utc_times = [startdate + timedelta(seconds=event) for event in newevent]
            # Keep track of combination with the most detected events
            if newevent.size>=100:
                info_lines.append(f"{name}, run 1")
            num_detections+=newevent.size
            # Save the cross-correlation values for each newevent
            cc_values = xcorrmean[newdect]
            #  Write detected events to output file
            with open(output_file_path, "a", encoding=("utf-8")) as output_file:
                if os.stat(output_file_path).st_size == 0:
                    output_file.write("starttime,templ,crosscorr value, run\n")
                for i, utc_time in enumerate(utc_times):
                    output_file.write(
                        f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                        f"{templ_idx},{cc_values[i]:.4f},1\n"
                    )

            # Reuse detected events as templates by stacking
            print("Got detections so let's reuse them as templates by stacking them!")
            print(len(utc_times))
            cpt=1
            for _ in range(num_repeats):
                cpt+=1
                if reuse_events:
                    # Plot new templates, cross-correlation, and new detections
                    all_template=[]
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
                            # max_abs_value = np.max(np.abs(tr.data[dect:dect + len(template)]))
                            stacked_templ[idx, :] += template.data #max_abs_value
                        # Cross-correlate stacked template with station data
                        xcorr_template = autocorr_tools.correlate_template(
                            tr.data, stacked_templ[idx,:],
                            mode='valid', normalize='full', demean=True, method='auto'
                        )
                        # Ensure equal length for cross-correlation arrays
                        if len(xcorr_template)<len(xcorr_full):
                            xcorr_full=xcorr_full[:len(xcorr_template)]
                        xcorr_full+=xcorr_template

                    all_template=stacked_templ/len(utc_times)

                    # Calculate mean cross-correlation
                    xcorrmean=xcorr_full/len(st)

                    # Plot stacked template on each station-channel combination
                    # TODO: no need to plot again the templates since it's the previous plot_stack?
                    template_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'template{cpt}', lastday)
                    crosscorr_tools.plot_template(st, all_template, pairs,
                                                  templ_idx, template_plot_filename)

                    # Find indices where the cross-correlation values are above the threshold
                    mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
                    thresh = 8 * mad
                    aboves = np.where(xcorrmean > thresh)

                   # Check if there are correlations above the threshold
                    if aboves[0].size > 0:
                        # Determine if there are new detections
                        windowlen = template.stats.npts
                        inds = aboves[0]
                        clusters = autocorr_tools.clusterdects(inds, windowlen)
                        newdect = autocorr_tools.culldects(inds, clusters, xcorrmean)

                        # Check if there are new detections
                        if newdect.size > 1:
                            print("Got new detections with the new templates!")
                            # Plot cross-correlation function for new detections
                            crosscorr_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'crosscorr{cpt}', lastday)
                            crosscorr_tools.plot_crosscorr(st, xcorrmean, thresh, newdect,
                                                            max_index, name,
                                                            lastday, crosscorr_plot_filename)

                            # Plot stacked traces for new detections
                            stack_plot_filename = crosscorr_tools.build_file_path(base_dir, folder, name, f'stack{cpt}', lastday)
                            crosscorr_tools.plot_stacks(st, template, newdect, pairs,
                                                        templ_idx, stack_plot_filename, cpt=cpt)

                            ## Writing in output.txt
                            # Create UTCDateTime objects from the newevent values
                            newevent = newdect*dt
                            utc_times = [startdate + timedelta(seconds=event) for event in newevent]
                            print(len(utc_times))
                            # Save the cross-correlation values for each newevent
                            cc_values = xcorrmean[newdect]
                            # Keep track of combination with the most detected events
                            if newevent.size>=100:
                                info_lines.append(f"{name}, run {cpt}")
                            num_detections+=newevent.size
                            #  Write the newevent and additional columns to the output file
                            with open(output_file_path, "a", encoding=("utf-8")) as output_file:
                                for i, utc_time in enumerate(utc_times):
                                    output_file.write(
                                        f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                                        f"{templ_idx},{cc_values[i]:.4f},{cpt}\n"
                                    )

    # Follow the advancement
    print(f"Processed template {templ_idx + 1}/{len(templates)}")

# Calculate and print script execution time
end_script = time.time()
script_execution_time = end_script - startscript
print(f"Script execution time: {script_execution_time:.2f} seconds")
# Get the list of stations and channels used
pairs_used = ", ".join(pairs)
# Write the info of the run in the output file
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {startdate} - {enddate}\n\n")
    file.write(f"Stations and Channels Used: {pairs_used}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write(f"Total of detections: {num_detections} (with redundant times)\n")
    file.write("More than 100 detections:\n")
    file.write("\n".join(info_lines) + '\n\n')
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')
