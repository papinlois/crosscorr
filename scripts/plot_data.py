#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:10:18 2024

@author: lpapin
"""

import os
import csv
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import crosscorr_tools

# Define the base directory
base_dir = "/Users/lpapin/Documents/phd/"
# folder = "bostock"
folder = "tim"

# ========== Plotting Data Streams ==========

# Define the network configurations # or network_configurations.py with import
## Bostock
# network_config = {
#     '1CN': {
#         'stations': ['LZB','PGC'],#, 'NLLB', 'SNB'], , 'VGZ'#
#         'channels': ['BHN', 'BHE', 'BHZ'],
#         'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
#     },
#     '2CN': {
#         'stations': ['PFB', 'YOUB'], #
#         'channels': ['HHN', 'HHE', 'HHZ'],
#         'filename_pattern': '{date}.CN.{station}..{channel}.mseed'
#     },
#     'C8': {
#         'stations': ['MGCB', 'JRBC'], # , 'PHYB', 'SHVB','LCBC', 'GLBC', 'TWBB'
#         'channels': ['HHN', 'HHE', 'HHZ'],
#         'filename_pattern': '{date}.C8.{station}..{channel}.mseed'
#     },
#     'PO': {
#         'stations': ['TSJB', 'SILB', 'SSIB', 'KLNB'], # , 'TWKB'
#         'channels': ['HHN', 'HHE', 'HHZ'],
#         'filename_pattern': '{date}.PO.{station}..{channel}.mseed'
#     }
# }
# date_of_interests = ["20050903","20050904","20050905","20050906","20050907","20050908",
#                         "20050909","20050910","20050911","20050912","20050913","20050914",
#                         "20050915","20050916","20050917","20050918","20050919","20050920",
#                         "20050921","20050922","20050923","20050924","20050925"]

## Tim
from network_configurations import network_config
# Days of data
startdate = datetime.strptime("20100504", "%Y%m%d")
enddate = datetime.strptime("20100520", "%Y%m%d")
date_of_interests = []
current_date = startdate
while current_date <= enddate:
    date_of_interests.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)
lastday=date_of_interests[-1] #For filenames

# Frequency range, sampling_rate, and time window
freqmin = 1.0
freqmax = 8.0
sampling_rate = 40.0

# Get the streams and preprocess ###actualize the get_traces fct
st = crosscorr_tools.get_traces(network_config, date_of_interests, base_dir)
st = crosscorr_tools.process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax)

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

# ========== Plotting Stations with Events ==========
# Either LFE families from Bostock or detections from Tim

plt.figure(figsize=(10, 8))

##Tim
startdate = datetime.strptime("20100504", "%Y%m%d")
enddate = datetime.strptime("20100520", "%Y%m%d")
# Load LFE data on Tim's catalog
templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
templates=templates[(templates['residual']<0.5)]
templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
templates = templates[(templates['OT'] >= startdate)
                    & (templates['OT'] < enddate)
                    & (templates['residual'] < 0.1)]
templates = templates.drop(columns=['dates', 'residual', 'starttime','dt'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
templates = templates.sort_values(by='N', ascending=False)
print(templates)

events = templates[['lon', 'lat', 'depth', 'OT']]

day_of_month = templates['OT'].dt.day
cmap = plt.cm.viridis
plt.scatter(events['lon'], events['lat'], c='grey', marker='x', label='Events')
# sc = plt.scatter(events['lon'], events['lat'], c=templates['OT'], cmap=cmap, marker='x', label='Events')
# cbar = plt.colorbar(sc)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14) 
plt.title('Events with Day of the Month Colorbar', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

## Stations
# Define the file path
file_path = "stations.csv"

# Open the file and read its contents
with open(file_path, newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile, delimiter='|')
    # Skip the header row
    next(reader)
    # Initialize empty lists to store data
    latitudes = []
    longitudes = []
    networks = []
    stations = []
    # Iterate over each row in the CSV file
    for row in reader:
        # Extract the desired fields
        network = row[0]
        station = row[1]
        latitude = float(row[2])
        longitude = float(row[3])
        # Append data to lists
        networks.append(network)
        stations.append(station)
        latitudes.append(latitude)
        longitudes.append(longitude)

# Create a color map based on unique networks using the 'tab10' colormap
unique_networks = list(set(networks))
color_map = matplotlib.colormaps['tab10']

# Plot each station with a triangle marker based on the network
for i, network in enumerate(unique_networks):
    network_indices = [j for j, net in enumerate(networks) if net == network]
    network_latitudes = [latitudes[k] for k in network_indices]
    network_longitudes = [longitudes[k] for k in network_indices]
    plt.scatter(network_longitudes, network_latitudes, 
                color=color_map(i), marker='^', s=150) #, label=network

# Annotate each station with its name
for i, (lon, lat, station) in enumerate(zip(longitudes, latitudes, stations)):
    plt.text(lon, lat, station, fontsize=12, ha='left')
'''
## Bostock
# Load LFE family data
sav_family_phases = np.load('./sav_family_phases.npy', allow_pickle=True).item()
family_nb = np.array(list(sav_family_phases.keys()))
family_nb = family_nb[family_nb == '052']

# Iterate through LFE families
for nb in family_nb:
    eqLoc_values = sav_family_phases[nb]['eqLoc']
    lon_family, lat_family, z_family = eqLoc_values
    lon_family = lon_family * -1
    plt.scatter(lon_family, lat_family, color='grey', s=75)
    plt.text(lon_family, lat_family, nb, fontsize=10, ha='left')

# Set the limits of the plot
plt.xlim(-124.5, -123)
plt.ylim(48, 49.3)

# Add labels and legend
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.title('Stations and LFE Families', fontsize=16)
plt.gcf().subplots_adjust(bottom=0.2)
plt.tight_layout()
# plt.legend()
plt.savefig('stations_lfe_families')
plt.show()
'''
