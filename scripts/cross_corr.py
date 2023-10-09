# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:56:50 2023

@author: papin
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#!pip install obspy #Google Colab
from obspy.core import Stream, read
from obspy.signal.cross_correlation import correlate_template
import autocorr_tools
import matplotlib.colorbar as clrbar
from scipy import stats
from scipy.signal import correlate

# Start timer
startscript = time.time()

# Read station locations from CSV file
locfile = pd.read_csv('stations.csv')
locs = locfile[['Name', 'Longitude', 'Latitude']].values

# # Plot station locations
# plt.figure()
# plt.plot(locfile['Longitude'], locfile['Latitude'], 'bo')
# for name, lon, lat in locs:
#     plt.text(lon, lat, name)
# plt.savefig('C:/Users/papin/Desktop/phd/plots/station_locations.png')
# plt.close()

# Create an empty Stream object
st = Stream()

# List of stations to analyze
stas = ['LZB','SNB']#'PGC','NLLB']
# stas = ['B010']#,'B926']


# List of channels to read
channels = ['BHE']#,'BHN','BHZ']
# channels = ['EH2']#,'EH1','EHZ']
#if multiple channel used, cross-correlation has to be modified so it doesn't do it with the station itself

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
st.trim(starttime=start + 21 * 3600, endtime=start + 21 * 3600 + 3600, nearest_sample=True, pad=True, fill_value=0)
st.detrend(type='simple')
st.filter("bandpass", freqmin=1.0, freqmax=10.0)

# # Add locations
# for ii in range(0,len(stas)):
#     ind=np.where(locs[:,0]==stas[ii])
#     st[ii].stats.y=locs[ind,1][0][0]
#     st[ii].stats.x=locs[ind,2][0][0]

# Plot the data
plt.figure()
offset = 0
for tmp in range(len(st)):
    shade = tmp / len(st)  # Adjust this value to control the shade (0 = dark blue, 1 = light blue)
    color = (0, 0, 0.5 + shade / 2)  # Adjust the values to control the shade
    plt.plot(st[tmp].times("timestamp"), st[tmp].data / np.max(np.abs(st[tmp].data)) + offset, color=color)#, label=stas[tmp])
    offset += 1
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Normalized Data + Offset', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.savefig('C:/Users/papin/Desktop/phd/plots/data_plot.png')
plt.close()

# Cross-correlation parameters
windowdur = 6  # Template window duration in seconds
windowlen = int(windowdur * st[0].stats.sampling_rate)  # Template window length in points
windowstep = 3  # Time shift for next window in seconds
windowsteplen = int(windowstep * st[0].stats.sampling_rate)  # Time shift in points
numwindows = int((st[0].stats.npts - windowlen) / windowsteplen)  # Number of time windows in interval
xcorrmean = np.zeros((numwindows, st[0].stats.npts - windowlen + 1))

# # Cross-correlation for 1 set of data
# for ii, tr in enumerate(st):
#     xcorrfull = np.zeros((numwindows, tr.stats.npts - windowlen + 1))
#     for kk in range(numwindows):
#         xcorrfull[kk, :] = autocorr_tools.correlate_template(tr.data, tr.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)],
#                                                               mode='valid', normalize='full', demean=True, method='auto')
#     xcorrmean += xcorrfull

# Cross-correlation between different stations
xcorrmean = np.zeros((numwindows, st[0].stats.npts - windowlen + 1))
for i in range(len(st)):
    for j in range(i + 1, len(st)):
        tr1 = st[i]
        tr2 = st[j]
        xcorrfull = np.zeros((numwindows, tr1.stats.npts - windowlen + 1))
        for kk in range(numwindows):
            cc1=xcorrfull[kk, :] = autocorr_tools.correlate_template(tr1.data, tr2.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)],
                                                                  mode='valid', normalize='full', demean=True, method='auto')
            #[(kk * windowsteplen):(kk * windowsteplen + windowlen)] when using as a template
            # cc2=correlate_template(tr1.data, tr2.data[(kk * windowsteplen):(kk * windowsteplen + windowlen)])
            # print(cc1-cc2)
        xcorrmean += xcorrfull
        
        # Network autocorrelation
        xcorrmean = xcorrmean / len(st)
        
        # Median absolute deviation
        mad = np.median(np.abs(xcorrmean - np.median(xcorrmean)))  # Median absolute deviation
        print(8*mad)
        thresh = 8
        print(np.max(xcorrmean))
        aboves = np.where(xcorrmean > thresh * mad)
        
        if aboves[0].size == 0:
            print("No significant correlations found.")
        else:
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
            plt.savefig('C:/Users/papin/Desktop/phd/plots/detection_plot.png')
            plt.close()
        
            # Plot the cross-correlation function ###amt (with template)
            winind = stats.mode(aboves[0])[0][0] # Most common value (template)
            xcorr = xcorrmean[winind, :]
            fig, ax = plt.subplots(figsize=(10, 3))
            t = st[0].stats.delta * np.arange(len(xcorr))
            ax.plot(t, xcorr)
            ax.axhline(thresh * mad, color='red')
            # inds = np.where(xcorr > thresh * mad)[0]
            # clusters = autocorr_tools.clusterdects(inds, windowlen)
            # newdect = autocorr_tools.culldects(inds, clusters, xcorr)
            # ax.plot(newdect * st[0].stats.delta, xcorr[newdect], 'kx')
            # ax.text(60, 1.1 * thresh * mad, '8*MAD', fontsize=16, color='red')
            ax.set_xlabel('Seconds of Hour 21 on 18/5', fontsize=14)
            ax.set_ylabel('Correlation Coefficient', fontsize=14)
            ax.set_xlim((0, 3600))
            plt.gcf().subplots_adjust(bottom=0.2)
            plt.savefig('C:/Users/papin/Desktop/phd/plots/correlation_function_plot.png')
            plt.close()
            
# Calculate and print script execution time
end_script = time.time()
print(f"Script execution time: {end_script - startscript:.2f} seconds")
