# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin
"""

import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.core import Stream, read
import autocorr_tools
import crosscorr_tools

# Plot station locations
locfile = pd.read_csv('stations.csv')
locs = locfile[['Name', 'Longitude', 'Latitude']].values
# crosscorr_tools.plot_station_locations(locs)

# Start timer
startscript = time.time()

# Initialize the list of CSV file paths
csv_file_paths = []

# List of stations/channels to analyze
# CN network
stas = ['NLLB','SNB']#,'PGC','LZB']
# stas=['YOUB','PFB']
channels = ['BHE']#,'BHN']#,'BHZ'] ###doesn't work for more than 1 channel
# # PB network
# stas = ['B010']#,'B926']
# channels = ['EH2']#,'EH1','EHZ']

# Hour and date of interest
hour_of_interest = 21
date_of_interest = "20100518"
date_to_find = f"{date_of_interest[:4]}-{date_of_interest[4:6]}-{date_of_interest[6:]}"

# Create CSV file paths for each station
for sta in stas:
    file_path = f'data/cut/cut_daily_CN.{sta}.csv'
    csv_file_paths.append(file_path)

# Call the function to merge and process the CSV data
result_df = crosscorr_tools.merge_csv_data(csv_file_paths, 
                                           date_to_find, hour_of_interest)

def load_station_data(stas, channels, date_of_interest):
    st = Stream()
    for sta in stas:
        for cha in channels:
            path = "C:/Users/papin/Desktop/phd/data/seed"
            file = f"{path}/{date_of_interest}.CN.{sta}..{cha}.mseed"
            try:
                tr = read(file)[0]  # Read only the first trace from the file
                st.append(tr)
                print("Loaded data:", tr)
            except FileNotFoundError:
                print(f"File {file} not found.")

    # Preprocessing: Interpolation, trimming, detrending, and filtering
    start = st[0].stats.starttime
    end = st[0].stats.endtime
    for tr in st:
        start = max(start, tr.stats.starttime)
        end = min(end, tr.stats.endtime)
    st.interpolate(sampling_rate=80, starttime=start)
    st.trim(starttime=start + hour_of_interest * 3600, 
            endtime=start + hour_of_interest * 3600 + 3600,
            nearest_sample=True, pad=True, fill_value=0)
    st.detrend(type='simple')
    st.filter("bandpass", freqmin=1.0, freqmax=10.0)

    return st

# Call the function to load station data
st = load_station_data(stas, channels, date_of_interest)

# Plot the data
plt.figure(figsize=(15, 5))
offset = 0
for sta_idx, sta in enumerate(stas):
    for cha_idx, cha in enumerate(channels):
        shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
        color = (0, 0, 0.5 + shade / 2)
        tr = st[sta_idx * len(channels) + cha_idx]
        plt.plot(tr.times("timestamp"), tr.data / np.max(np.abs(tr.data)) + offset,
                  color=color, label=f"{sta}_{cha}")
        offset += 1
        combo = f"Station: {sta}, Channel: {cha}"
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Normalized Data + Offset', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True)
plt.savefig('C:/Users/papin/Desktop/phd/plots/data_plot.png')
plt.show()

# Define the path to save threshold and xcorr values later
file_path = "C:/Users/papin/Desktop/phd/results.txt"

# Function to append values to the file
def append_to_file(filename, thresh_mad, max_xcorrmean):
    with open(filename, "a", encoding='utf-8') as file:
        file.write(f"{thresh_mad}\t{max_xcorrmean}\n")

# Check if the file already exists, if not, create a new one with headers
if not os.path.isfile(file_path):
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Thresh * Mad\tMax Xcorrmean\n")

# Cross-correlation parameters
tr=st[0]
windowdur  = 6   # Template window duration in seconds
windowstep = 3 # Time shift for next window in seconds
windowlen     = int(windowdur * tr.stats.sampling_rate)   # Template window length in points
windowsteplen = int(windowstep * tr.stats.sampling_rate)  # Time shift in points
numwindows = int((tr.stats.npts - windowlen) / windowsteplen)  # Number of time windows in interval

# Cross-correlation between different stations
xcorrmean = np.zeros((numwindows, tr.stats.npts - windowlen + 1))

for idx, row in result_df.iterrows():
    starttime = row['starttime']

    # Extract the date and time components
    date, time = starttime.split('T')

    # Convert date and time to datetime objects
    date_obj = pd.to_datetime(date)
    time_obj = pd.to_datetime(time)
    
    # Combine date and time to create a datetime object
    datetime_for_xcorr = date_obj.replace(
        hour=time_obj.hour,
        minute=time_obj.minute,
        second=time_obj.second,
    )
    
    ###datetime is the time of the event detected in cut : need to be used
    
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
            mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))
            thresh = 8
            aboves = np.where(xcorrmean > thresh * mad)
            
            # Append the values to the file
            append_to_file(file_path, thresh * mad, np.max(xcorrmean))
            
            # Construct a filename based on station combinations
            station_combination = f"{tr1.stats.station}_{tr2.stats.station}_{cha}"
            print(station_combination)
            
            if aboves[0].size == 0:
                print("No significant correlations found.")
            else:
                # Creation of the detection plot with template/time index
                detection_plot_filename = (
                    f'C:/Users/papin/Desktop/phd/plots/'
                    f'detection_plot_{station_combination}.png'
                )
                crosscorr_tools.create_detection_plot(aboves, xcorrmean, 
                                                      detection_plot_filename)
            
                # Creation of the cross-correlation plot
                correlation_function_plot_filename = (
                    f'C:/Users/papin/Desktop/phd/plots/'
                    f'correlation_function_plot_{station_combination}.png'
                )
                crosscorr_tools.plot_cross_correlation(xcorrmean, aboves, 
                                                       thresh, mad, windowlen, 
                                                       st, hour_of_interest, 
                                                       date_of_interest, 
                                                       correlation_function_plot_filename)

# Calculate and print script execution time
end_script = time.time()
print(f"Script execution time: {end_script - startscript:.2f} seconds")

# Create a histogram
plt.hist(aboves[1], bins=100)  # Adjust the number of bins as needed
plt.xlabel('Time Index')
plt.ylabel('Frequency')
plt.title('Histogram of Time Indices for Significant Correlations')
plt.show()

# Read data from the results.txt file
with open('C:/Users/papin/Desktop/phd/results.txt', 'r', encoding='utf-8') as file:
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

# Set the x-axis and y-axis parameters
plt.xticks(range(1, len(x_values) + 1))
plt.yticks([i * 0.1 for i in range(6)] + [1])
max_y = math.ceil(max(max(thresh_mad_values), 
                      max(max_xcorrmean_values)) / 0.1) * 0.1 + 0.1
plt.ylim(0, max_y)
plt.grid(axis='x')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.xlabel('Number')
plt.ylabel('Values of Correlation')
plt.legend()

# Show the plot
plt.show()
