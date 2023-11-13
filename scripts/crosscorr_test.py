# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:28:22 2023

This script is a simplified version of crosscorr.py that is intended to provide
a fast look at the crosscorrelation between a template and a stream, as well as their data.

@author: papin
"""

import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import read
import autocorr_tools
import crosscorr_tools

# Start timer
startscript = time.time()

# Parameters
sta = ['POHA']
cha = ['BH1']
network = 'IU'
date_of_interest = "20100729"

# Load and preprocess data
startdate = datetime.strptime(date_of_interest, "%Y%m%d")
enddate = startdate + timedelta(days=1)
st = read('POHA..BH1.2000-07-29-00-00-00.ms')
st.trim(starttime=UTCDateTime(2000, 7, 29, 8, 0, 0), endtime=UTCDateTime(2000, 7, 29, 9, 0, 0))  # Added endtime argument
st.interpolate(sampling_rate=20, starttime=st[0].stats.starttime)
st.detrend(type='simple')
st.filter("bandpass", freqmin=2, freqmax=7)

# Plot full stream
plt.figure(figsize=(15, 5))
plt.plot(st[0].times('matplotlib'), st[0].data, label=f"{sta}_{cha}")
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Ampl', fontsize=14)
plt.title('Stream', fontsize=16)
plt.grid(True)
plt.xlim(st[0].times('matplotlib').min(), st[0].times('matplotlib').max())  # Set xlim to min and max of the data
plt.savefig('crosscorr_test_mine1')
plt.show()

# Define templates
template1_start = UTCDateTime(2000, 7, 29, 8, 27, 36)
template1_end = UTCDateTime(2000, 7, 29, 8, 27, 45)
template1 = st.copy().trim(starttime=template1_start, endtime=template1_end)

template2_start = UTCDateTime(2000, 7, 29, 8, 57, 26.5)
template2_end = UTCDateTime(2000, 7, 29, 8, 57, 35.5)
template2 = st.copy().trim(starttime=template2_start, endtime=template2_end)

# Which one
template = template1

# Plot template
plt.figure(figsize=(15, 5))
plt.plot(template[0].times('matplotlib'), template[0].data, label='Template')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Ampl', fontsize=14)
plt.title('Template', fontsize=16)
plt.grid(True)
plt.show()

# Cross-correlation
xcorr_template = autocorr_tools.correlate_template(st[0].data, template[0].data, mode='valid', normalize='full', demean=True, method='auto')

# Detection and plotting
thresh = 0.8
aboves = np.where(xcorr_template > thresh)

stream_duration = st[0].stats.endtime - st[0].stats.starttime
windowlen = template[0].stats.npts
inds = np.where(xcorr_template > thresh)[0]
clusters = autocorr_tools.clusterdects(inds, windowlen)
newdect = autocorr_tools.culldects(inds, clusters, xcorr_template)
max_index = np.argmax(xcorr_template[newdect])

if newdect.size > 1:
    fig, ax = plt.subplots(figsize=(15, 5))
    t = st[0].stats.delta * np.arange(len(xcorr_template))
    ax.plot(t, xcorr_template)
    ax.axhline(thresh, color='red')
    ax.plot(newdect * st[0].stats.delta, xcorr_template[newdect], 'kx')
    ax.plot((newdect * st[0].stats.delta)[max_index], (xcorr_template[newdect])[max_index], 'gx', markersize=10, linewidth=10)
    ax.text(60, 1.1 * thresh, '0.8', fontsize=14, color='red')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Coefficient', fontsize=14)
    ax.set_xlim(0, stream_duration)
    plt.title('Cross-Correlation', fontsize=16)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig('crosscorr_test_mine2')
    plt.show()
