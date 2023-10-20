# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

Functions are from autocorrelation and crosscorrelation tools

@author: papin
"""

import os
import time
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.core import Stream, read
import autocorr_tools
import crosscorr_tools

# Plot station locations
locfile = pd.read_csv('stations.csv')
locs = locfile[['Name', 'Longitude', 'Latitude']].values
crosscorr_tools.plot_station_locations(locs)

# Start timer
startscript = time.time()

# List of stations/channels to analyze
# CN network
stas = ['LZB','SNB','PGC','NLLB']
channels = ['BHE']

# Hour and date of interest
date_of_interest = "20100516"
startdate=datetime.strptime(date_of_interest, "%Y%m%d")
enddate=startdate+timedelta(days=1)

st = Stream()
for sta in stas:
    for cha in channels:
        path = "C:/Users/papin/Desktop/phd/data/seed"
        file = f"{path}/{date_of_interest}.CN.{sta}..{cha}.mseed"
        try:
            tr = read(file)[0]  # Careful for other networks
            st.append(tr)
            #print("Loaded data:", tr)
        except FileNotFoundError:
            print(f"File {file} not found.")

# Preprocessing: Interpolation, trimming, detrending, and filtering
start = st[0].stats.starttime
end = st[0].stats.endtime
for tr in st:
    start = max(start, tr.stats.starttime)
    end = min(end, tr.stats.endtime)
st.interpolate(sampling_rate=80, starttime=start) # Can be modified
st.trim(starttime=start,endtime=end, # Can be modified
        nearest_sample=True, pad=True, fill_value=0)
st.detrend(type='simple')
st.filter("bandpass", freqmin=1.0, freqmax=10.0) # Can be modified

# Add locations
for ii, sta in enumerate(stas):
    ind = np.where(locs[:, 0] == sta)
    st[ii].stats.y = locs[ind, 1][0][0]
    st[ii].stats.x = locs[ind, 2][0][0]

# Call the function to plot the data
cha=crosscorr_tools.plot_data(st, stas, channels)

# Define the path to save threshold and xcorr values later
file_path = "C:/Users/papin/Desktop/phd/threshold.txt"

# Function to append values to the file
def append_to_file(filename, thresh_mad, max_xcorr):
    """
    Append threshold and maximum cross-correlation values to a text file.
    """
    with open(filename, "a", encoding='utf-8') as file:
        file.write(f"{thresh_mad}\t{max_xcorr}\n")

# Check if the file already exists, if not, create a new one with headers
if not os.path.isfile(file_path):
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Thresh * Mad\tMax xcorr\n")

# load LFE data
df_full=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
df_full=df_full[(df_full['residual']<0.5)] # & (df_full['N']>3)]
df_full['datetime']=pd.to_datetime(df_full['OT'])
templates=df_full[(df_full['datetime']>=startdate) & (df_full['datetime']<enddate) & (df_full['residual']<0.1)]

# Add a new column for the index of the lines
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'

# Collect information
combinations = []  # Store significant correlations
info_lines = []  # Store lines of information

# Get the list of stations used
stations_used = ", ".join(stas)

# Iterate over all stations
for i in range(len(st)):
    tr1 = st[i]
    
    # Iterate over all templates
    for idx in range(len(templates)):
        template_stats = templates.iloc[idx]
        start = UTCDateTime(template_stats['datetime'] + timedelta(seconds=10))
        end = UTCDateTime(template_stats['datetime'] + timedelta(seconds=40))
        template = st.copy().trim(starttime=start, endtime=end)
        
        # Initialize xcorr for the current station and template
        xcorr = np.zeros(tr1.stats.npts - template[0].stats.npts + 1)

        # Calculate cross-correlation using the current template
        for j in range(len(template)):
            tr2 = template[j]
            xcorr_template = autocorr_tools.correlate_template(
                tr1.data,tr2.data,
                mode='valid', normalize='full', demean=True, method='auto'
            )
            xcorr += xcorr_template
        
        # Find indices where the cross-correlation values are above the threshold
        mad = np.median(np.abs(xcorr - np.median(xcorr)))
        thresh = 8
        aboves = np.where(xcorr > thresh * mad)

        # Extract station and template information
        station = tr1.stats.station
        channel = tr1.stats.channel
        template_index = idx

        # Construct a filename based on station combinations and template_index
        crosscorr_combination = (
            f'{station}_{channel}_template{template_index}'
        )
        #print(crosscorr_combination)
        if aboves[0].size == 0:
            #print("No significant correlations found")
            info_lines.append(f"{crosscorr_combination}:"
                              f" No significant correlations found")
        else:
            # Creation of the cross-correlation plot
            correlation_plot_filename = (
                f'C:/Users/papin/Desktop/phd/plots/'
                f'correlation_function_plot_{crosscorr_combination}.png'
            )
            windowlen=template[0].stats.npts
            fig, ax = plt.subplots(figsize=(10,3))
            t=st[0].stats.delta*np.arange(len(xcorr))
            ax.plot(t,xcorr)
            ax.axhline(thresh*mad,color='red')
            inds=np.where(xcorr>thresh*mad)[0]
            clusters=autocorr_tools.clusterdects(inds,windowlen)
            newdect=autocorr_tools.culldects(inds,clusters,xcorr)
            ax.plot(newdect*st[0].stats.delta,xcorr[newdect],'kx')
            ax.text(60,1.1*thresh*mad,'8*MAD',fontsize=16,color='red')
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Correlation Coefficient', fontsize=14) ###
            ax.set_xlim((0,86000))
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.savefig(correlation_plot_filename)
            plt.close()

# Calculate and print script execution time
end_script = time.time()
script_execution_time = end_script - startscript
print(f"Script execution time: {script_execution_time:.2f} seconds")

# Write information to a text file
info_file_path = "C:/Users/papin/Desktop/phd/info.txt"

with open(info_file_path, 'w', encoding='utf-8') as file:
    file.write(f"Date Range: {startdate} - {enddate}\n\n")
    file.write(f"Stations and Channel Used: {stations_used} --- {channel}\n\n")
    file.write("Templates:\n")
    file.write(templates.to_string() + '\n\n')
    file.write("No significant Correlations:\n")
    file.write("\n".join(info_lines) + '\n\n')
    file.write(f"Script execution time: {script_execution_time:.2f} seconds\n")
  
# Read data from the results.txt file
with open('C:/Users/papin/Desktop/phd/threshold.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    thresh_mad_values = []
    max_xcorr_values = []
    for line in lines[1:]:  # Skip the header line
        parts = line.split('\t')
        thresh_mad_values.append(float(parts[0]))
        max_xcorr_values.append(float(parts[1]))

# Create the scatter plot
x_values = range(1, len(thresh_mad_values) + 1)  # Generate line numbers
plt.scatter(x_values, thresh_mad_values, c='blue', label='Thresh * Mad')
plt.scatter(x_values, max_xcorr_values, c='red', label='Max Xcorr')

# Set the x-axis and y-axis parameters
plt.xticks(range(1, len(x_values) + 1))
plt.yticks([i * 0.1 for i in range(6)] + [1])
max_y = math.ceil(max(max(thresh_mad_values), 
                      max(max_xcorr_values)) / 0.1) * 0.1 + 0.1
plt.ylim(0, max_y)
plt.grid(axis='x')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.xlabel('Number')
plt.ylabel('Values of Correlation')
plt.legend()

# Show the plot
plt.show()
