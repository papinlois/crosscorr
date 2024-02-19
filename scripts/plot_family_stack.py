#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:54:44 2024

This script is made for stacking seismic signals for each station/channel
combination. In the crosscorr work, it stacks detections from the same family
for multiple stations over a number of days. Can be applied to other catalogs.

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

# ========== Loading Data ==========

# Define the network configurations #see crosscorr_bostock
network_config = {
    '1CN': {
        'stations': ['LZB'],#,'PGC'],#, 'NLLB', 'SNB'], , 'VGZ'#
        'channels': ['BHN', 'BHE', 'BHZ'],
        'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    # },
    # '2CN': {
    #     'stations': ['PFB'],#, 'YOUB'], #
    #     'channels': ['HHN', 'HHE', 'HHZ'],
    #     'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
    # },
    # 'C8': {
    #     'stations': ['MGCB', 'JRBC'], # , 'PHYB', 'SHVB','LCBC', 'GLBC', 'TWBB'
    #     'channels': ['HHN', 'HHE', 'HHZ'],
    #     'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
    # },
    # 'PO': {
    #     'stations': ['SILB', 'SSIB', 'KLNB'], # , 'TSJB', 'TWKB'
    #     'channels': ['HHN', 'HHE', 'HHZ'],
    #     'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
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
dt = 1/sampling_rate
win_size = 50

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, sampling_rate, freqmin, freqmax)

# List of stations/channels to analyze
pairs=[]
for tr in st:
    iid = tr.id
    identifier = iid[3:]
    pairs.append(identifier)

# Remove the bad data
tr_remove = ['PGC..BHE', 'SHVB..HHE', 'SHVB..HHN','SNB..BHE','TWKB..HHE','VGZ..BHE','YOUB..HHZ']
idx_remove = [i for i, tr in enumerate(st)
              if tr.stats.station + '..' + tr.stats.channel in tr_remove]
if idx_remove:
    for idx in sorted(idx_remove, reverse=True):
        st.pop(idx)
        del pairs[idx]

# Load LFE data on Bostock's catalog
templates = pd.read_csv('lfe_svi.txt', index_col=0, dtype={'date': str, 'hour': str, 'lfe_family':str})
templates['date'] = '20' + templates['date']
templates['date'] = pd.to_datetime(templates['date'], format='%Y%m%d')
templates['hour'] = templates['hour'].str.zfill(2)
templates['OT'] = templates['date'] + pd.to_timedelta(templates['hour'].astype(int), unit='h') + pd.to_timedelta(templates['second'], unit='s')
templates = templates[(templates['OT'] >= startdate) & (templates['OT'] < enddate)]
templates = templates.drop(columns=['Mw','hour','second','date'])
templates.reset_index(inplace=True)
templates.index.name = 'Index'
## To choose which templates
templ_idx='002'
templates = templates[templates['lfe_family'] == templ_idx]
print(templates)
# Select 20 templates randomly from the last set of templates
selected_indices = random.sample(range(len(templates)), 50)
templates = templates.iloc[selected_indices]
print(templates)

# ========== Plot Section ==========

stacked_traces = np.zeros((len(st), win_size * int(sampling_rate) + 1))
OT_templ = templates['OT']

for idx, tr in enumerate(st):
    for start_templ in OT_templ:
        start_templ = UTCDateTime(start_templ) #+ timedelta(seconds=win_size)
        end_templ = start_templ + timedelta(seconds=50)
        template = tr.copy().trim(starttime=start_templ, endtime=end_templ)
        # Normalize the template waveform
        max_amplitude = np.max(np.abs(template.data))
        if max_amplitude != 0:
            template.data /= max_amplitude
        # Add the normalized template to the stacked trace
        stacked_traces[idx, :] += template.data

# Normalize the stacked traces by the number of templates
stacked_traces /= len(OT_templ)

plt.figure(figsize=(12,6))
nb = 1  # Distance between plots
offset = len(stacked_traces) * nb
x = np.linspace(0, len(template) / st[0].stats.sampling_rate,
                len(stacked_traces[0, :]), endpoint=False)
for i in range(len(stacked_traces)):
    norm = np.max(np.abs(stacked_traces[i,:]))
    plt.plot(x, stacked_traces[i, :] / norm + offset, label=f'{pairs[i]}')
    offset -= nb
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Normalized Data + Offset', fontsize=14)
plt.title(f'Stacked Traces for Template {templ_idx}', fontsize=16)
plt.yticks(np.arange(len(pairs)) * nb+nb, pairs[::-1], fontsize=12)
# plt.xlim(26,30)
plt.ylim(0, len(pairs) * nb + nb)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'plots', f"{folder}",f'templ_{templ_idx}_example_stack_2005'))
plt.show()
