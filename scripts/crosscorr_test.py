# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:28:22 2023

This script is a simplified version of crosscorr.py that is intended to provide
a fast look at the crosscorrelation between a template and a stream, as well as their data.

@author: papin
"""

import time
from datetime import timedelta
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import read
from obspy import Stream
from matplotlib.gridspec import GridSpec

# Start timer
startscript = time.time()

# Load and preprocess data
st=Stream()
cha='EH2'
if cha=='EH1':
    comp=0
elif cha=='EH2':
    comp=1
elif cha=='EHZ':
    comp=2

st = read('C:/Users/papin/Desktop/phd/data/seed/B009.PB.2010.137')
st.filter("bandpass", freqmin=2, freqmax=8)
st.merge()
# st.detrend(type='simple')
# st.interpolate(sampling_rate=80, starttime=st[comp].stats.starttime)

# Template 46
template=st.copy().trim(starttime=UTCDateTime(2010, 5, 17, 1, 54, 44),endtime=UTCDateTime(2010, 5, 17, 1, 55, 14))
st.trim(starttime=UTCDateTime(2010, 5, 17, 13, 0, 0), endtime=UTCDateTime(2010, 5, 17, 14, 0, 0))

# Get some detections for examples
detection1_start = UTCDateTime(2010, 5, 17, 13, 9, 34)
# detection1_start = UTCDateTime(2010, 5, 17, 3, 43, 56)
detection1_end = detection1_start + timedelta(seconds=30)
detection1 = st.copy().trim(starttime=detection1_start, endtime=detection1_end)

detection2_start = UTCDateTime(2010, 5, 17, 13, 43, 59)
# detection2_start = UTCDateTime(2010, 5, 17, 7, 5, 54)
detection2_end = detection2_start + timedelta(seconds=30)
detection2 = st.copy().trim(starttime=detection2_start, endtime=detection2_end)

detection3_start = UTCDateTime(2010, 5, 17, 13, 51, 53)
# detection3_start = UTCDateTime(2010, 5, 17, 10, 30, 57)
detection3_end = detection3_start + timedelta(seconds=30)
detection3 = st.copy().trim(starttime=detection3_start, endtime=detection3_end)

# detection4_start = UTCDateTime(2010, 5, 17, 13, 43, 59)
# detection4_end = detection4_start + timedelta(seconds=30)
# detection4 = st.copy().trim(starttime=detection4_start, endtime=detection4_end)

# detection5_start = UTCDateTime(2010, 5, 17, 15, 44, 52)
# detection5_end = detection5_start + timedelta(seconds=30)
# detection5 = st.copy().trim(starttime=detection5_start, endtime=detection5_end)

# detection6_start = UTCDateTime(2010, 5, 17, 22, 17, 35)
# detection6_end = detection6_start + timedelta(seconds=30)
# detection6 = st.copy().trim(starttime=detection6_start, endtime=detection6_end)

# Create the figure
fig = plt.figure(constrained_layout=True,figsize=(18,8))
fig.suptitle('Seismic Data and Detections', fontsize=16)  # Add this line for the figure title
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])
# ax4 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])
# ax6 = fig.add_subplot(gs[1, 2])

# Plot detections
ax1.plot(detection1[comp].times('relative'),detection1[comp].data,'tab:red')
ax1.set_xlim((detection1[comp].times('relative')[0],detection1[comp].times('relative')[-1]))
ax1.set_xlabel('Time(s)',fontsize=14)
ax1.set_ylabel('Counts',fontsize=14)

ax2.plot(detection2[comp].times('relative'),detection2[comp].data,'tab:green')
ax2.set_xlim((detection2[comp].times('relative')[0],detection2[comp].times('relative')[-1]))
ax2.set_xlabel('Time(s)',fontsize=14)
ax2.set_ylabel('Counts',fontsize=14)

ax3.plot(detection3[comp].times('relative'),detection3[comp].data,'tab:orange')
ax3.set_xlim((detection3[comp].times('relative')[0],detection3[comp].times('relative')[-1]))
ax3.set_xlabel('Time(s)',fontsize=14)
ax3.set_ylabel('Counts',fontsize=14)

# Raw data with highlights of detections
ax4.plot_date(st[comp].times('matplotlib'),st[comp].data,'tab:grey',xdate=True)
ax4.plot_date(detection1[comp].times('matplotlib'),detection1[comp].data,'tab:red',xdate=True)
ax4.plot_date(detection2[comp].times('matplotlib'),detection2[comp].data,'tab:green',xdate=True)
ax4.plot_date(detection3[comp].times('matplotlib'),detection3[comp].data,'tab:orange',xdate=True)
ax4.set_xlim((st[comp].times('matplotlib')[0],st[comp].times('matplotlib')[-1]))
ax4.set_xlabel('Time(s)',fontsize=14)
ax4.set_ylabel('Counts',fontsize=14)

# # Plot detections
# ax4.plot(detection4[comp].times('relative'),detection4[comp].data,'tab:blue')
# ax4.set_xlim((detection4[comp].times('relative')[0],detection4[comp].times('relative')[-1]))
# ax4.set_xlabel('Time(s)',fontsize=14)
# ax4.set_ylabel('Counts',fontsize=14)

# ax5.plot(detection5[comp].times('relative'),detection5[comp].data,'tab:brown')
# ax5.set_xlim((detection5[comp].times('relative')[0],detection5[comp].times('relative')[-1]))
# ax5.set_xlabel('Time(s)',fontsize=14)
# ax5.set_ylabel('Counts',fontsize=14)

# ax6.plot(detection6[comp].times('relative'),detection6[comp].data,'tab:purple')
# ax6.set_xlim((detection6[comp].times('relative')[0],detection6[comp].times('relative')[-1]))
# ax6.set_xlabel('Time(s)',fontsize=14)
# ax6.set_ylabel('Counts',fontsize=14)
