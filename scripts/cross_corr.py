# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin
"""

import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.colorbar as clrbar
import numpy as np
import pandas as pd
#!pip install obspy #Google Colab
from obspy.core import Stream, read
from obspy.signal.cross_correlation import correlate_template
import autocorr_tools
from scipy import stats
from scipy.signal import correlate

# # Function to plot station locations
# def plot_station_locations(locs):
#     plt.figure()
#     plt.plot(locs[:, 1], locs[:, 2], 'bo')
#     for name, lon, lat in locs:
#         plt.text(lon, lat, name)
#     plt.savefig('C:/Users/papin/Desktop/phd/plots/station_locations.png')
#     plt.close()

# Plot station locations
locfile = pd.read_csv('stations.csv')
locs = locfile[['Name', 'Longitude', 'Latitude']].values
# plot_station_locations(locs)

# Start timer
startscript = time.time()

# Create an empty Stream object
st = Stream()

# List of stations to analyze
stas = ['LZB','SNB','PGC','NLLB']
# stas = ['B010']#,'B926']

# List of channels to read
channels = ['BHE','BHN','BHZ']
# channels = ['EH2']#,'EH1','EHZ']

# Load data for selected stations
for sta in stas:
    for cha in channels:
        path = "C:/Users/papin/Desktop/phd/data"
        file = f"{path}/20100518.CN.{sta}..{cha}.mseed"
        # file = f"{path}/{sta}.PB.2010.138"
        #file=f"20100516.CN.{sta}..BHE.mseed" #Google Colab
        try:
            tr = read(file)[0]  # Read only the first trace from the file
            # tr = read(file)[1]
            print(tr)
            st.append(tr)
            print(f"Loaded data from {file}")
        except FileNotFoundError:
            print(f"File {file} not found.")

# Preprocessing: Interpolation, trimming, detrending, and filtering
start = st[0].stats.starttime
end = st[0].stats.endtime
for tr in st:
    start = max(start, tr.stats.starttime)
    end = min(end, tr.stats.endtime)

# Trim all traces to the same time window
st.interpolate(sampling_rate=80, starttime=start)
st.trim(starttime=start + 21 * 3600, endtime=start + 21 * 3600 + 3600,
        nearest_sample=True, pad=True, fill_value=0)
st.detrend(type='simple')
st.filter("bandpass", freqmin=1.0, freqmax=10.0)

# Add locations
for ii, sta in enumerate(stas):
    ind = np.where(locs[:, 0] == sta)
    st[ii].stats.y = locs[ind, 1][0][0]
    st[ii].stats.x = locs[ind, 2][0][0]

# Plot the data
plt.figure(figsize=(15, 5))
offset = 0
for sta_idx, sta in enumerate(stas):
    for cha_idx, cha in enumerate(channels):
        tr = st[sta_idx * len(channels) + cha_idx]  # Access the trace based on station and channel index
        shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
        color = (0, 0, 0.5 + shade / 2)
        plt.plot(tr.times("timestamp"), tr.data / np.max(np.abs(tr.data)) + offset,
                 color=color, label=f"{sta}_{cha}")
        offset += 1
        combo = f"Station: {sta}, Channel: {cha}"  # Combination of station and channel as a string
        print("Station-Channel Combination:", combo)  # Print the combination
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Normalized Data + Offset', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
# plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
# current_xlim = plt.xlim()
# tick_positions, tick_labels = plt.xticks()
# plt.xlim(1274217000,1274218000)
plt.grid(True)
plt.savefig('C:/Users/papin/Desktop/phd/plots/data_plot.png')
plt.show()

# Define the path for the file
file_path = "C:/Users/papin/Desktop/phd/results.txt"

# Check if the file already exists, if not, create a new one with headers
if not os.path.isfile(file_path):
    with open(file_path, "w") as file:
        file.write("Thresh * Mad\tMax Xcorrmean\n")

# Function to append values to the file
def append_to_file(filename, thresh_mad, max_xcorrmean):
    with open(filename, "a") as file:
        file.write(f"{thresh_mad}\t{max_xcorrmean}\n")

# Cross-correlation parameters
tr=st[0]
windowdur  = 30  # Template window duration in seconds
windowstep = 2.5  # Time shift for next window in seconds
windowlen     = int(windowdur * tr.stats.sampling_rate)   # Template window length in points
windowsteplen = int(windowstep * tr.stats.sampling_rate)  # Time shift in points
numwindows = int((tr.stats.npts - windowlen) / windowsteplen)  # Number of time windows in interval

# Cross-correlation between different stations
xcorrmean = np.zeros((numwindows, tr.stats.npts - windowlen + 1))
for i in range(len(st)):
    tr1 = st[i]
    for j in range(i + 1, len(st)):
        tr2 = st[j]
        xcorrfull = np.zeros((numwindows, tr1.stats.npts - windowlen + 1))
        # Calculate cross-correlation
        for kk in range(numwindows):
            xcorrfull[kk, :] = autocorr_tools.correlate_template(
                tr1.data, tr2.data[kk * windowsteplen : (kk * windowsteplen + windowlen)],
                mode='valid', normalize='full', demean=True, method='auto'
            )
        xcorrmean += xcorrfull
        
        # Network autocorrelation
        xcorrmean /= len(st)
        
        # Median absolute deviation
        mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))  # Median absolute deviation
        thresh = 8
        aboves = np.where(xcorrmean > thresh * mad)
        
        # Append the values to the file
        append_to_file(file_path, thresh * mad, np.max(xcorrmean))

        if aboves[0].size == 0:
            print("No significant correlations found.")
        else:
            # Construct a filename based on station combinations
            station_combination = f"{tr1.stats.station}_{tr2.stats.station}"

            # Save the figures with a specific name
            detection_plot_filename = f'C:/Users/papin/Desktop/phd/plots/detection_plot_{station_combination}.png'
            correlation_function_plot_filename = f'C:/Users/papin/Desktop/phd/plots/correlation_function_plot_{station_combination}.png'
            
            # Detection plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(aboves[0], aboves[1], s=20, c=xcorrmean[aboves])
            ax.set_xlabel('Template Index', fontsize=14)
            ax.set_ylabel('Time Index', fontsize=14)
            cax, _ = clrbar.make_axes(ax)
            cbar = clrbar.ColorbarBase(cax)
            cbar.ax.set_ylabel('Correlation Coefficient', rotation=270, labelpad=15, fontsize=14)
            ax.set_xlim((np.min(aboves[0]), np.max(aboves[0])))
            ax.set_ylim((np.min(aboves[1]), np.max(aboves[1])))
            plt.savefig(detection_plot_filename)
            plt.close()
        
            # Plot the cross-correlation function ###amt (with template)
            winind = stats.mode(aboves[0])[0][0] # Most common value (template)
            xcorr = xcorrmean[winind, :]
            fig, ax = plt.subplots(figsize=(10, 3))
            t = st[0].stats.delta * np.arange(len(xcorr)) ###need to be modified
            ax.plot(t, xcorr)
            ax.axhline(thresh * mad, color='red')
            inds = np.where(xcorr > thresh * mad)[0]
            clusters = autocorr_tools.clusterdects(inds, windowlen)
            newdect = autocorr_tools.culldects(inds, clusters, xcorr)
            ax.plot(newdect * st[0].stats.delta, xcorr[newdect], 'kx')
            ax.text(60, 1.1 * thresh * mad, '8*MAD', fontsize=16, color='red')
            ax.set_xlabel('Seconds of Hour 21 on 18/5', fontsize=14)
            ax.set_ylabel('Correlation Coefficient', fontsize=14)
            ax.set_xlim((0, 3600))
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.savefig(correlation_function_plot_filename)
            plt.close()
    
# Calculate and print script execution time
end_script = time.time()
print(f"Script execution time: {end_script - startscript:.2f} seconds")

# Read data from the results.txt file
with open('C:/Users/papin/Desktop/phd/results.txt', 'r') as file:
    lines = file.readlines()
    thresh_mad_values = []
    max_xcorrmean_values = []

    for line in lines[1:]:  # Skip the header line
        parts = line.split('\t')
        thresh_mad_values.append(float(parts[0]))
        max_xcorrmean_values.append(float(parts[1]))

# Create the scatter plot
x_values = range(1, len(thresh_mad_values) + 1)  # Generate line numbers
plt.scatter(x_values, thresh_mad_values, c='blue', label='Thresh * Mad')
plt.scatter(x_values, max_xcorrmean_values, c='red', label='Max Xcorrmean')

# Set the y-axis limits, ticks, and grid
plt.yticks([i * 0.2 for i in range(6)] + [1])  # Custom ticks every 0.2 and the maximum 1
plt.ylim(0, 0.8)
plt.grid(axis='y', linestyle='--', linewidth=0.5)  # Add a grid with dashes

# Set the x-axis limits and grid
plt.xticks(range(1, len(x_values) + 1))
plt.grid(axis='x')

# Add labels and legend
plt.xlabel('Line Number')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

