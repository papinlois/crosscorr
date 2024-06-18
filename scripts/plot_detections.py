#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:26:34 2024

Plot the original detections and the highest cc coeff of the new ones.

@author: lpapin

As of 17/06/24.
"""

## Detections
import pandas as pd
detections=pd.read_csv('/Users/lpapin/Desktop/SSE_2005/aug_PO_2/yes/output_aug_PO_yes.txt')
detections=detections.sort_values(['starttime'])
print(detections)
# # Filtering only "high" cc value
# coeff_thresh=0.5
# filtered_detections = detections[detections['coeff'] > coeff_thresh]
# print(filtered_detections)
# detections=filtered_detections
# Filtering specific template
specific_template = 4
filtered_detections = detections[(detections['template'] == specific_template)]
print(filtered_detections)
detections=filtered_detections

# Getting the template
import numpy as np
cc_coeffmax=np.max(detections['coeff'])
original_event = detections[(detections['coeff'] == cc_coeffmax)]
print(original_event)
# Filtering time period
from datetime import datetime
startdate = datetime.strptime("20050913", "%Y%m%d")
enddate = datetime.strptime("20050916", "%Y%m%d")
detections['starttime'] = pd.to_datetime(detections['starttime'])
starttime = datetime.combine(startdate, datetime.strptime("00:00:00", "%H:%M:%S").time())
endtime = datetime.combine(enddate, datetime.strptime("23:59:00", "%H:%M:%S").time())
filtered_detections = detections[(detections['starttime'] >= starttime) & (detections['starttime'] < endtime)]
print(filtered_detections)
detections=filtered_detections

## Streams
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime, timedelta
import crosscorr_tools

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
# Define the network
from network_configurations import network_config #for now for PO
# Days of data
dates_of_interest = []
current_date = startdate
while current_date <= enddate:# + timedelta(days=1):
    dates_of_interest.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)
lastday = dates_of_interest[-1] #For filenames
# Get the streams
st, pairs = crosscorr_tools.get_traces(network_config, dates_of_interest, base_dir)
# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 10
st = crosscorr_tools.process_data(st, startdate, enddate+timedelta(days=1), sampling_rate, freqmin, freqmax)
print(st.__str__(extended=True))

## Plotting
N=4 # Only highest N values
detections=detections.sort_values(['coeff'])[::-1]
detections.reset_index(inplace=True, drop=True)
detections.index.name = 'Index'
print(detections)
if detections['coeff'][0]==cc_coeffmax:
    detections=detections[1:]
    detections.reset_index(inplace=True, drop=True)
import matplotlib.pyplot as plt
from obspy import UTCDateTime
for tr in st:
    fig, axs = plt.subplots(N+2, figsize=(10, 20))
    fig.suptitle(tr.id, fontsize=16)
    # Plot original event
    original_event_time = original_event['starttime'].iloc[0]
    start_templ = UTCDateTime(original_event_time)
    end_templ = start_templ + timedelta(seconds=win_size)
    original_event_data = tr.copy().trim(starttime=start_templ, endtime=end_templ)
    axs[0].plot(original_event_data.times(), original_event_data.data, color='orange', linewidth=2)
    axs[0].set_title('Original Event: {}'.format(start_templ))
    axs[0].grid(True)
    # Plot detections
    stacked_data = np.zeros(len(original_event_data.data))
    for i, detection in detections[:N].iterrows():
        detection_time = detection['starttime']
        start_det = UTCDateTime(detection_time)
        end_det = start_det + timedelta(seconds=win_size)
        detection_data = tr.copy().trim(starttime=start_det, endtime=end_det)
        axs[i+1].plot(detection_data.times(), detection_data.data, linewidth=1)
        axs[i+1].set_title('Detection {}: {} - cc={:.2f}'.format(i+1, start_det, detection['coeff']))
        axs[i+1].grid(True)
        stacked_data += detection_data.data
    # Plot stacked waveform for the first N detections
    axs[N+1].plot(original_event_data.times(), stacked_data, color='green', linewidth=1)
    axs[N+1].set_title(f'Stacked {N} Detections')
    axs[N+1].grid(True)
    plt.tight_layout()
    plt.show()

print("!!! pb de output in 'a' mode")
print("!!! the original event may be not the highest cc value")

##### Expensive
