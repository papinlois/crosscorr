# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

Functions are from autocorrelation and crosscorrelation tools

@author: papin
"""

import os
import time
import math
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
# crosscorr_tools.plot_station_locations(locs)

# Start timer
startscript = time.time()

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

# Initialize the list of CSV file paths
csv_file_paths = []

# Create CSV file paths for each station
for sta in stas:
    file_path = f'data/cut/cut_daily_CN.{sta}.csv'
    csv_file_paths.append(file_path)

# Call the function to merge and process the CSV data
result_df = crosscorr_tools.merge_csv_data(csv_file_paths, 
                                           date_to_find, hour_of_interest)

def load_station_data(stas, channels, date_of_interest):
    """
    Load and preprocess station data from MiniSEED files.
    """
    st = Stream()
    for sta in stas:
        for cha in channels:
            path = "C:/Users/papin/Desktop/phd/data/seed"
            file = f"{path}/{date_of_interest}.CN.{sta}..{cha}.mseed"
            try:
                tr = read(file)[0]  # Careful for other networks
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
    st.interpolate(sampling_rate=80, starttime=start) # Can be changed
    st.trim(starttime=start + hour_of_interest * 3600,
            endtime=start + hour_of_interest * 3600 + 3600, # Can be changed
            nearest_sample=True, pad=True, fill_value=0)
    st.detrend(type='simple')
    st.filter("bandpass", freqmin=1.0, freqmax=10.0) # Can be changed

    return st

# Call the function to load station data
st = load_station_data(stas, channels, date_of_interest)

# Call the function to plot the data
cha=crosscorr_tools.plot_data(st, stas, channels)

# Define the path to save threshold and xcorr values later
file_path = "C:/Users/papin/Desktop/phd/results.txt"

# Function to append values to the file
def append_to_file(filename, thresh_mad, max_xcorrmean):
    """
    Append threshold and maximum cross-correlation values to a text file.
    """
    with open(filename, "a", encoding='utf-8') as file:
        file.write(f"{thresh_mad}\t{max_xcorrmean}\n")

# Check if the file already exists, if not, create a new one with headers
if not os.path.isfile(file_path):
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Thresh * Mad\tMax Xcorrmean\n")

# Cross-correlation parameters
tr=st[0] # Can be changed
windowdur  = 30  # Template window duration in seconds
windowstep = 15  # Time shift for next window in seconds
windowlen     = int(windowdur * tr.stats.sampling_rate)   # In points
windowsteplen = int(windowstep * tr.stats.sampling_rate)  # In points
numwindows = int((tr.stats.npts - windowlen) / windowsteplen)  # Number of time windows in interval

# Cross-correlation between different stations
xcorrmean = np.zeros((numwindows, tr.stats.npts - windowlen + 1))

for idx, row in result_df.iterrows():
    starttime = row['starttime']

    # Extract the date and time components
    datee, timee = starttime.split('T')

    # Convert date and time to datetime objects
    date_obj = pd.to_datetime(datee)
    time_obj = pd.to_datetime(timee)
    
    # Combine date and time to create a datetime object
    datetime_for_xcorr = date_obj.replace(
        hour=time_obj.hour,
        minute=time_obj.minute,
        second=time_obj.second,
    )
    
    # Calculate the new template time (date_for_xcorr + 10 seconds)
    template_time = UTCDateTime(datetime_for_xcorr + pd.Timedelta(seconds=10))
    
    for i in range(len(st)):
        tr1 = st[i]
        print(tr1)
        # Find the template index in the trace
        template_index = int((template_time - tr1.stats.starttime) * tr1.stats.sampling_rate)            
        for j in range(i + 1, len(st)):
            tr2 = st[j]
            print(tr2)
            xcorrfull = np.zeros((numwindows, tr1.stats.npts - windowlen + 1))
            # Calculate cross-correlation using the template
            for k in range(numwindows):
                xcorrfull[k, :] = autocorr_tools.correlate_template(
                    tr1.data,
                    tr2.data[template_index:template_index + windowlen],  # Use the template data
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
            
            # Construct a filename based on station combinations and template_index
            station_combination = ( 
                f'{tr1.stats.station}_{tr2.stats.station}'
                f'_{cha}_template{template_index}'
            )
            
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

# # Create a histogram
# plt.hist(aboves[1], bins=100)  # Adjust the number of bins as needed
# plt.xlabel('Time Index')
# plt.ylabel('Frequency')
# plt.title('Histogram of Time Indices for Significant Correlations')
# plt.show()

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
