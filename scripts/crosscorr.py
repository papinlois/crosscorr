# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin

This script performs a network cross-correlation analysis on seismic data with
already known events as templates, in order to detect new events.
Functions are from autocorrelation and cross-correlation tools modules.

Description (not specifically in order):
1. Creates an info.txt file with relevant information about the data.
2. Loads seismic data from specified stations and channels for a given date.
3. Preprocesses the data by trimming, interpolating, detrending, and filtering.
4. Plots the full streams of seismic data.
5. Defines templates for cross-correlation analysis by using already known events.
6. Plots the station locations as well as the templates locations (optional).
7. Computes cross-correlation coefficients between each station's data and the template.
8. Detects significant correlations based on a specified threshold.
9. Generates and saves plots for cross-correlation analysis with detected events.
10. Outputs the new detected events times to a text file.
11. Stack and plot all detected events of each template.

As of 11/27/23.
"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import Stream
import autocorr_tools
import crosscorr_tools

# matplotlib.use('Agg') # Must fix the memory issue on Spyder
# matplotlib.use('TkAgg') # Interactively

# Define the base directory
base_dir = "C:/Users/papin/Desktop/phd/"

# Plot station locations
locfile = pd.read_csv(os.path.join(base_dir, 'stations.csv'))
locs = locfile[['Name', 'Longitude', 'Latitude','Network']].values

# Start timer
startscript = time.time()

# List of stations/channels to analyze : CN network
stas = ['LZB','SNB','PGC','NLLB']
channels = ['BHN','BHE','BHZ']
pairs = [f"{sta}..{cha}" for sta in stas for cha in channels]
channel_prefix = channels[0][:2]
network = 'CN'

# Hour and date of interest
date_of_interest = "20100517"
startdate=datetime.strptime(date_of_interest, "%Y%m%d")
enddate=startdate+timedelta(days=1)

# Frequency range, sampling_rate, and time window
freqmin = 2.0
freqmax = 8.0
sampling_rate = 40.0
win_size = 30

# Get the streams and preprocess
st = Stream(traces=crosscorr_tools.get_traces(pairs, date_of_interest, base_dir))
st = crosscorr_tools.process_data(st, sampling_rate, freqmin, freqmax)

# Plot all the streams
crosscorr_tools.plot_data(st, stas, channels, base_dir)

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

# # Extract relevant columns for events
# events = templates[['lon', 'lat', 'depth', 'datetime']]
# crosscorr_tools.plot_locations(locs, base_dir, events=events)

# Collect information
info_lines = []  # Store lines of information
num_detections = 0

# Generate the output files paths
folder = f"{network} {channel_prefix} {date_of_interest}" #To create beforehand
info_file_path = os.path.join(base_dir, 'plots', f"{folder}", "info.txt")
output_file_path = os.path.join(base_dir, 'plots', f"{folder}", 'output.txt')

# Process templates in batches (memory reason)
template_groups = [templates[i:i + 20] for i in range(0, len(templates), 20)]
for batch_idx, template_group in enumerate(template_groups):
    # Iterate over all templates
    for idx, template_stats in template_group.iterrows():
        # Iterate over all stations and channels combination
        master_xcorr=np.zeros(int(st[0].stats.npts-(win_size*sampling_rate)))
        for tr in st:
            # Template data
            start_templ = UTCDateTime(template_stats['datetime'] + timedelta(seconds=10))
            end_templ = start_templ + timedelta(seconds=win_size)
            template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
            
            # FIXME: For NLLB where values of 0 exist, a Warning appears for
            # the line 470 in autocorr_tools.py. Issue with the np.sqrt in
            # correlate_template function but still running
            # Cross-correlation function
            xcorr_template = autocorr_tools.correlate_template(
                tr.data, template.data,
                mode='valid', normalize='full', demean=True, method='auto'
            )
            master_xcorr+=xcorr_template
        
        # Network cross-correlation
        xcorrmean=master_xcorr/len(st)
        # FIXME: Do we put the np.mean?
        # xcorrmean=np.mean(np.vstack(xcorr_template),axis=0)

        del xcorr_template, master_xcorr

        # Find indices where the cross-correlation values are above the threshold
        mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
        thresh = 8 * mad
        aboves = np.where(xcorrmean > thresh)

        # Construct a filename
        template_index = idx
        iid = tr.get_id()[3:]
        crosscorr_combination = f'{iid}_templ{template_index}'

        # Calculate the duration of the data in seconds for the plot
        stream_duration = (st[0].stats.endtime - st[0].stats.starttime)

        # Template does match at least once (cc value of 1)
        if aboves[0].size > 0:
            # Calculate the window length for clustering
            windowlen = template.stats.npts
            # Indices where the cross-correlation values are above the threshold
            inds = aboves[0]
            # Cluster the detected events
            clusters = autocorr_tools.clusterdects(inds, windowlen)
            # Cull detections within clusters
            newdect = autocorr_tools.culldects(inds, clusters, xcorrmean)
            # Find the index of the maximum value in newdect
            max_index = np.argmax(xcorrmean[newdect])

            # Creation of the cross-correlation plot only if new events detected
            if newdect.size > 1:
                crosscorr_plot_filename = os.path.join(
                    base_dir,
                    f'plots/{folder}/templ{template_index}_crosscorr_{iid}_{date_of_interest}.png'
                )

                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(tr.stats.delta*np.arange(len(xcorrmean)),xcorrmean)
                ax.axhline(thresh,color='red')
                ax.plot(newdect*tr.stats.delta,xcorrmean[newdect],'kx')
                ax.plot((newdect*tr.stats.delta)[max_index],
                        (xcorrmean[newdect])[max_index],'gx', markersize=10, linewidth=10)
                ax.text(60,1.1*thresh,'8*MAD',fontsize=14,color='red')
                ax.set_xlabel('Time (s)', fontsize=14)
                ax.set_ylabel('Correlation Coefficient', fontsize=14)
                ax.set_xlim(0, stream_duration)
                ax.set_title(f'{crosscorr_combination} - {date_of_interest}', fontsize=16)
                plt.gcf().subplots_adjust(bottom=0.2)
                plt.savefig(crosscorr_plot_filename)
                plt.close()

                del fig, ax

                # Create UTCDateTime objects from the newevent values
                newevent = np.delete(newdect, max_index)*st[0].stats.delta
                utc_times = [st[0].stats.starttime.datetime +
                             timedelta(seconds=event) for event in newevent]
                
                # Keep track of combination with the most detected events
                if newevent.size>=100:
                    info_lines.append(f"{crosscorr_combination}")
                num_detections+=newevent.size

                # Save the cross-correlation values for each newevent
                mask = xcorrmean[newdect] != (xcorrmean[newdect])[max_index]
                cc_values = xcorrmean[newdect][mask]

                # Plot the stack associated to sta..cha
                crosscorr_tools.plot_stack(utc_times, cc_values, tr,
                                           win_size, template, template_index,
                                           iid, date_of_interest, base_dir)

                # Write the newevent and additional columns to the output file
                with open(output_file_path, "a", encoding=("utf-8")) as output_file:
                    if os.stat(output_file_path).st_size == 0:
                        output_file.write("starttime,templ,id,crosscorr value\n")
                    for i, utc_time in enumerate(utc_times):
                        output_file.write(
                            f"{UTCDateTime(utc_time).strftime('%Y-%m-%dT%H:%M:%S.%f')},"
                            f"{template_index},{iid},{cc_values[i]:.4f}\n"
                        )
            else:
                del xcorrmean

    # Follow the advancement
    print(f"Processed batch {batch_idx + 1}/{len(template_groups)}")
    break

# Calculate and print script execution time
end_script = time.time()
script_execution_time = end_script - startscript
print(f"Script execution time: {script_execution_time:.2f} seconds")
# Get the list of stations and channels used
stas_used = ", ".join(stas)
channels_used = ", ".join(channels)
# Write the info of the run in the output file
with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {startdate} - {enddate}\n\n")
    file.write(f"Stations and Channel Used: {stas_used} --- {channels_used}\n\n")
    file.write(f"Frequency range: {freqmin}-{freqmax} Hz\n")
    file.write(f"Sampling rate: {sampling_rate} Hz\n\n")
    file.write(f"Total of detections: {num_detections} \n")
    file.write("More than 100 detections:\n")
    file.write("\n".join(info_lines) + '\n\n')
    file.write("Templates info:\n")
    file.write(templates.to_string() + '\n')
