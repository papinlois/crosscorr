#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:41:31 2024

@author: lpapin
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import time
from datetime import datetime, timedelta
import crosscorr_tools

startscript = time.time()

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
folder = "tim"

# Define the network
from network_configurations import network_config

# Days of data
startdate = datetime.strptime("20050903", "%Y%m%d")
enddate = datetime.strptime("20050904", "%Y%m%d")
dates_of_interest = []
current_date = startdate
while current_date <= enddate:
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
# st = crosscorr_tools.process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax)
print(st.__str__(extended=True))

# List of stations/channels to analyze
pairs=[]
stas=[]
for tr in st:
    pairs.append(tr.id[3:])
    stas.append(tr.stats.station)
stas=list(set(stas))

# Plot the streams to see how the data looks like; help to understand the quality
# of the results and to remove bad data that are difficult to automatically
# remove with parameters in functions. NB: memory expensive (a lot!)
data_plot_filename = os.path.join(base_dir,f'plots/{folder}/data_plot.png')
crosscorr_tools.plot_data(st, pairs, data_plot_filename)
