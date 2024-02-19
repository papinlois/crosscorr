#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:10:18 2024

@author: lpapin
"""

import os
import random
from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
from obspy import UTCDateTime
import crosscorr_tools
import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
folder = "bostock"

# ========== Plotting Data Streams ==========

# Define the network configurations # or network_configurations.py with import
network_config = {
    '1CN': {
        'stations': ['LZB','PGC'],#, 'NLLB', 'SNB'], , 'VGZ'#
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    '2CN': {
        'stations': ['PFB', 'YOUB'], #
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    },
    'C8': {
        'stations': ['MGCB', 'JRBC'], # , 'PHYB', 'SHVB','LCBC', 'GLBC', 'TWBB'
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    },
    'PO': {
        'stations': ['TSJB', 'SILB', 'SSIB', 'KLNB'], # , 'TWKB'
        'channels': ['HHN', 'HHE', 'HHZ'],
        'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
    }
}

# Days of data
date_of_interests = ["20050903","20050904","20050905","20050906","20050907","20050908",
                        "20050909","20050910","20050911","20050912","20050913","20050914",
                        "20050915","20050916","20050917","20050918","20050919","20050920",
                        "20050921","20050922","20050923","20050924","20050925"]
startdate=datetime.strptime(date_of_interests[0], "%Y%m%d")
enddate=startdate+timedelta(days=len(date_of_interests) - 1)
lastday=date_of_interests[-1] #For filenames

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, sampling_rate, freqmin, freqmax)

# List of stations/channels to analyze
pairs=[]
for tr in st:
    iid = tr.id
    identifier = iid[3:]
    pairs.append(identifier)

# Plot all the streams
data_plot_filename = os.path.join(
    base_dir,
    f'plots/{folder}/data_plot.png'
)
crosscorr_tools.plot_data(st, pairs, data_plot_filename)

# ========== Plotting Stations ==========

# =============================================================================
# ## TO UPDATE TO MORE STATIONS AND NETWORK
# # Plot station locations
# locfile = pd.read_csv('stations.csv')
# locs = locfile[['Name', 'Longitude', 'Latitude','Network']].values
# crosscorr_tools.plot_loc(locs, base_dir)
# =============================================================================
