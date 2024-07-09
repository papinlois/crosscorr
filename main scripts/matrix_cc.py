#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon July 8 10:22:21 2024

This script creates the cross-correlation coefficients between every unique 
pair of events. This is aim to be used then for clustering with dendrogram.

NB: Made for 1 station at a time and comp Z is choosen.

@author: papin

As of 08/07/24.
"""

# ================ Initialization ================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from obspy import UTCDateTime
import autocorr_tools
import crosscorr_tools

startscript = time.time()

# Define the base directory
base_dir = "/home/lpapin/crosscorr"
folder = "SSE_2005"
which = 'talapas'
# Generate the output files paths

# Define the network
from network_configurations_aug_PO import network_config
stas = [station for value in network_config.values() for station in value['stations']]
network_config['PO']['stations'] = [stas[0]] #SILB ###

# ================ Events ================

# Load and preprocess detections
detections = pd.read_csv(base_dir+'/output_aug_PO_yes.txt')
detections = detections.sort_values(['coeff'], ascending=False)
print(detections)

# ================ Template-matching parameters ================

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0
dt = 1/sampling_rate
win_size = 10
tr_remove = ['PGC..BHE','SHVB..HHE','SHVB..HHN','SNB..BHE',
             'TWKB..HHE','VGZ..BHE','YOUB..HHZ']

# ================ Functions ================

def write_output(idx1, idx2, xcorr, time_event1, time_event2, output_file_path):
    """
    Write cross-correlation coefficient and additional columns to output file.
    """
    data = output_file_path[-16:-4] #!!For 4 characters stations
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        if os.stat(output_file_path).st_size == 0:
            output_file.write(f"idx1,idx2,{data},time_event1,time_event2\n")
        output_file.write(
            f"{idx1},{idx2},{xcorr:.2f},{time_event1},{time_event2}\n"
        )

# ================ Cross-correlation process ================

for idx1, detection1 in detections.iterrows():
    ## Time and data parameters
    time_event1 = UTCDateTime(detection1['starttime'])
    start_data1 = time_event1
    end_data1 = time_event1 + timedelta(seconds=win_size)

    # Get the streams and preprocess
    current_date1 = time_event1.strftime("%Y%m%d")
    dates_of_interest1 = [current_date1]
    st1, pairs1 = crosscorr_tools.get_traces(
        network_config, dates_of_interest1, base_dir, which=which)

    # Remove the bad data: has to be specfic to the period you're looking at
    st1, pairs1 = crosscorr_tools.remove_stations(st1, pairs1, tr_remove)

    # Process and choose the right stream
    st1 = crosscorr_tools.process_data(
        st1, start_data1, end_data1, sampling_rate, freqmin, freqmax)
    st1 = st1.select(channel="*Z")

    for idx2, detection2 in detections.iterrows():
        if idx2!=idx1:
            ## Time and data parameters
            time_event2 = UTCDateTime(detection2['starttime'])
            start_data2 = time_event2
            end_data2 = time_event2 + timedelta(seconds=win_size)

            # Get the streams and preprocess
            current_date2 = time_event2.strftime("%Y%m%d")
            dates_of_interest2 = [current_date2]
            st2, pairs2 = crosscorr_tools.get_traces(
                network_config, dates_of_interest2, base_dir, which=which)

            # Remove the bad data: has to be specfic to the period you're looking at
            st2, pairs2 = crosscorr_tools.remove_stations(st2, pairs2, tr_remove)

            # Process and choose the right stream
            st2 = crosscorr_tools.process_data(
                st2, start_data2, end_data2, sampling_rate, freqmin, freqmax)
            st2 = st2.select(channel="*Z")

            # When the data is not available still putting a value to it
            if not st1:
                # Write the crosscorr coeff and additional columns to the output file
                xcorr=np.nan
                data = st1[0].id
                output_file_path = os.path.join(base_dir, 'plots', f"{folder}/output_{data}.txt")
                write_output(idx1, idx2, xcorr[0], time_event1, time_event2, output_file_path)
                continue
            if not st2:
                # Write the crosscorr coeff and additional columns to the output file
                xcorr=np.nan
                data = st1[0].id
                output_file_path = os.path.join(base_dir, 'plots', f"{folder}/output_{data}.txt")
                write_output(idx1, idx2, xcorr[0], time_event1, time_event2, output_file_path)
                continue

            ## Cross-correlation process
            tr1=st1[0]
            tr2=st2[0]
            event1=tr1.copy().data
            event2=tr2.copy().data

            # Cut the same length for
            if len(event1) != len(event2):
                min_len = min(len(event1), len(event2))
                event1 = event1[:min_len]
                event2 = event2[:min_len]

            # Cross-correlate template with station data
            xcorr = autocorr_tools.correlate_template(
                event1, event2,
                mode='valid', normalize='full', demean=True, method='auto'
            )

            # Check if there are any NaN values and make it 0
            if np.isnan(xcorr).any():
                xcorr = np.nan_to_num(xcorr)
                print('NaN values replaced with 0')

            # Write the crosscorr coeff and additional columns to the output file
            data = st1[0].id
            output_file_path = os.path.join(base_dir, 'plots', f"{folder}", f"output_{data}.txt")
            write_output(idx1, idx2, xcorr[0], time_event1, time_event2, output_file_path)
