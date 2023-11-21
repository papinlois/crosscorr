# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:56:50 2023

@author: papin

This script performs a network cross-correlation analysis on seismic data with
already known events as templates, in order to detect new events.
Functions are from autocorrelation and cross-correlation tools modules.

Description (not specifically in order):
1. Creates an info.txt file with relevant information about the data.
2. Loads seismic data from specified stations and channels for a given date.
3. Preprocesses the data by trimming, interpolating, detrending, and filtering.
4. Plots the full streams of seismic data.
5. Defines templates for cross-correlation analysis.
6. Plots the station locations as well as the templates locations (optional).
7. Computes cross-correlation coefficients between each station's data and the template.
8. Detects significant correlations based on a specified threshold.
9. Generates and saves plots for cross-correlation analysis with detected events.
10. Outputs the new detected events times to a text file.
11. Stack and plot all detected events of each template.

PS: Make sure to adjust the input parameters such as station names, channels, date_of_interest, templates.
PSS: This script has been made for PB network for which the date format is different.

As of 11/21/23.
"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import Stream
import autocorr_tools
import crosscorr_tools

# Define the base directory
base_dir = "C:/Users/papin/Desktop/phd/"

# Plot station locations
locfile = pd.read_csv(os.path.join(base_dir, 'stations.csv'))
locs = locfile[['Name', 'Longitude', 'Latitude','Network']].values

# Start timer
startscript = time.time()

# List of stations/channels to analyze
stas = ['B001', 'B009', 'B010', 'B011', 'B926']
channels = ['EH1','EH2','EHZ']
channel_prefix = channels[0][:2]
network = 'PB'

# Hour and date of interest
date_of_interest = "20100518"
startdate = datetime.strptime(date_of_interest, "%Y%m%d")
enddate = startdate + timedelta(days=1)

# Frequency range and sampling_rate
freqmin = 2.0; freqmax = 8.0
sampling_rate = 80.0

# Plot the data as it is
crosscorr_tools.plot_data(date_of_interest, stas, channels, network, base_dir)

# Get the streams and preprocess
st = Stream(traces=crosscorr_tools.get_traces_PB(stas, channels, startdate, network, base_dir))
st, stas = crosscorr_tools.process_data(st, stas, locs=locs, sampling_rate=sampling_rate, freqmin=freqmin, freqmax=freqmax)

# Load LFE data on Tim's catalog
df_full=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
df_full=df_full[(df_full['residual']<0.5)] # & (df_full['N']>3)]
df_full['datetime']=pd.to_datetime(df_full['OT'])
templates = df_full[(df_full['datetime'] >= st[0].stats.starttime.datetime) 
                    & (df_full['datetime'] < st[0].stats.endtime.datetime) 
                    & (df_full['residual'] < 0.1)]
templates = templates.drop(columns=['dates', 'N'])
# Add a new column for the index of the lines (for output file)
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
del df_full # too big to keep and useless

# Extract relevant columns for events
events = templates[['lon', 'lat', 'depth', 'datetime']]
crosscorr_tools.plot_locations(locs, base_dir, events=events)

# Collect information
info_lines = []  # Store lines of information
output_file_path = os.path.join(base_dir, 'plots', f"{network} {channel_prefix} {date_of_interest}", 'output.txt')

# Process templates in batches of 20
template_groups = [templates[i:i + 20] for i in range(0, len(templates), 20)]
for batch_idx, template_group in enumerate(template_groups):
    try:
        # Iterate over all templates
        for idx, template_stats in template_group.iterrows():
            # Iterate over all stations and channels
            xcorr_templates = []
            for tr1 in st:
                template_stats = templates.iloc[idx]
                start_st = UTCDateTime(template_stats['datetime'] + timedelta(seconds=10))
                end_st = UTCDateTime(template_stats['datetime'] + timedelta(seconds=40))
                template = tr1.copy().trim(starttime=start_st, endtime=end_st)

                xcorr_template = autocorr_tools.correlate_template(
                    tr1.data, template.data,
                    mode='valid', normalize='full', demean=True, method='auto'
                )
                xcorr_templates.append(xcorr_template)

            xcorrfull = np.vstack(xcorr_templates)
            xcorrmean = np.mean(xcorrfull, axis=0)

            del xcorr_template, xcorr_templates, xcorrfull

            # Find indices where the cross-correlation values are above the threshold
            mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
            thresh = 8
            aboves = np.where(xcorrmean > thresh * mad)

            # Construct a filename
            template_index = idx
            crosscorr_combination = f'net{network}_cha{channel_prefix}_templ{template_index}'

            # Calculate the duration of the data in seconds for the plot
            stream_duration = (st[0].stats.endtime - st[0].stats.starttime)

            if aboves[0].size == 0:
                info_lines.append(f"{crosscorr_combination}:"
                                  f" No significant correlations found")
            else:
                correlation_plot_filename = os.path.join(base_dir, f"plots/crosscorr_{crosscorr_combination}_{date_of_interest}.png")

                # Calculate the window length for clustering
                windowlen = template.stats.npts
                # Find indices where the cross-correlation values are above the threshold
                inds = np.where(xcorrmean > thresh * mad)[0]
                # Cluster the detected events
                clusters = autocorr_tools.clusterdects(inds, windowlen)
                # Cull detections within clusters
                newdect = autocorr_tools.culldects(inds, clusters, xcorrmean)
                # Find the index of the maximum value in newdect
                max_index = np.argmax(xcorrmean[newdect])
                if newdect.size > 1:
                    fig, ax = plt.subplots(figsize=(10, 3))
                    t = st[0].stats.delta * np.arange(len(xcorrmean))
                    ax.plot(t, xcorrmean)
                    ax.axhline(thresh * mad, color='red')
                    ax.plot(newdect * st[0].stats.delta, xcorrmean[newdect], 'kx')
                    ax.plot((newdect * st[0].stats.delta)[max_index],
                            (xcorrmean[newdect])[max_index], 'gx', markersize=10, linewidth=10)
                    ax.text(60, 1.1 * thresh * mad, '8*MAD', fontsize=14, color='red')
                    ax.set_xlabel('Time (s)', fontsize=14)
                    ax.set_ylabel('Correlation Coefficient', fontsize=14)
                    ax.set_xlim(0, stream_duration)
                    ax.set_title(f'{crosscorr_combination} - {date_of_interest}', fontsize=16)
                    plt.gcf().subplots_adjust(bottom=0.2)
                    plt.savefig(correlation_plot_filename)
                    plt.close()
                    del t, fig, ax

                    # Create UTCDateTime objects from the newevent values
                    newevent = np.delete(newdect, max_index)*st[0].stats.delta
                    utc_times = [st[0].stats.starttime.datetime + timedelta(seconds=event) for event in newevent]

                    # Save the cross-correlation values for each newevent
                    mask = xcorrmean[newdect] != (xcorrmean[newdect])[max_index]
                    cc_values = xcorrmean[newdect][mask]

                    # Write the newevent and additional columns to the output file
                    with open(output_file_path, "a") as output_file:
                        if os.stat(output_file_path).st_size == 0:
                            output_file.write("starttime,templ,channel,network,crosscorr value\n")
                        for i, utc_time in enumerate(utc_times):
                            output_file.write(
                                f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                                f"{idx},{channel_prefix},{network},{cc_values[i]:.4f}\n"
                            )
                else:
                    del xcorrmean

        # Follow the advancement
        print(f"Processed batch {batch_idx + 1}/{len(template_groups)}")

    except MemoryError as me:
        print(f"MemoryError occurred: {me} at template {idx}.")
    except Exception as e:
        print(f"An error occurred processing template {idx}: {e}")

    finally:
        # Calculate and print script execution time
        end_script = time.time()
        script_execution_time = end_script - startscript
        print(f"Script execution time: {script_execution_time:.2f} seconds")
        # Get the list of stations and channels used
        stations_used = ", ".join(stas)
        channels_used = ", ".join(channels)
        # Create the info.txt file with relevant information
        info_file_path = os.path.join(base_dir, 'plots', f"{network} {channel_prefix} {date_of_interest}", "info.txt")
        with open(info_file_path, 'w', encoding='utf-8') as file:
            file.write(f"Date Range: {startdate} - {enddate}\n\n")
            file.write(f"Stations and Channel Used: {stations_used} --- {channels_used}\n\n")
            file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
            file.write(f"Sampling rate of {sampling_rate} Hz\n\n")
            file.write(f"Script execution time: {script_execution_time:.2f} seconds\n\n")
            file.write("Templates:\n")
            file.write(templates.to_string() + '\n\n')
            file.write("No significant Correlations:\n") # if there is an issue and even the event template is not correlated
            file.write("\n".join(info_lines) + '\n\n')

    # break # 1st template group

# Plot and save all summed traces on the new detected events time
window_size = 30
templates = templates.loc[[69]] # Specific template for the comparision
crosscorr_tools.plot_summed_traces_PB(stas, channels, window_size, network, startdate, date_of_interest, base_dir, templates)
