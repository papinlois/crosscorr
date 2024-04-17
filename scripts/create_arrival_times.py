#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:09:14 2024

@author: papin

This script creates a .txt file that relates estimated times for P- and 
S-waves for each of Tim's detections + some figures to show the data.

As of 04/04/24.
"""

import os
import csv
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import crosscorr_tools

base_dir = "/Users/lpapin/Documents/phd/"

def idx_loc(loc):
    """
    Given a rough location, return the index of the closest loc in grid-nodes.
    """
    d2 = np.sum((coords[:, :3] - loc)**2, axis=1)
    idx = np.argmin(d2)
    return idx

# Load the grid for computing the times
# A = np.load('./all_T_0.1_3.npy', allow_pickle=True).item() # Which stations detected the events
TT = np.load('Travel.npy', allow_pickle=True).item()
# sav_family_phases = np.load('./sav_family_phases.npy', allow_pickle=True).item()

# Load detections for Tim's catalog
startdate = datetime.strptime("20050903", "%Y%m%d")
enddate = datetime.strptime("20050925", "%Y%m%d")
# Load LFE data on Tim's catalog
templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
templates=templates[(templates['residual']<0.5)]
templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
templates = templates[(templates['OT'] >= startdate)
                    & (templates['OT'] < enddate)
                    & (templates['residual'] < 0.1)]
templates = templates.sort_values(by='starttime', ascending=True)
templates = templates.drop(columns=['dates', 'residual','dt','OT'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
print(templates)

# Extract necessary data
coords = np.array(list(TT['T'].keys())) # Coordinates of each points on the grid
sta_phase = TT['sta_phase'] # Station and P/S waves list
# family_nb = np.array(list(sav_family_phases.keys()))
# output_file_path = os.path.join(base_dir, 'arrival_times_bostock.txt')
output_file_path = os.path.join(base_dir, 'arrival_times_tim_2005_SSE.txt')

# Check if the output file already exists
if not os.path.isfile(output_file_path):
    coords2 = np.zeros((len(templates), 3)) # Coordinates of each detection
    for i, row in templates.iterrows():
        coords2[i] = [row['lon'], row['lat'], row['depth']]
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Write header if the file is empty
        output_file.write("template,station,P-wave,S-wave,difference,lon_event,lat_event,z_event\n")
    for idx in range(len(templates)):
        lon_event, lat_event, z_event = coords2[idx]
        exact_loc = coords[idx_loc([lon_event, lat_event, z_event])]
        arriv_times = TT['T'][tuple(exact_loc)]
        # Write data for each station and its P- and S-wave arrival times
        for i in range(0, int(len(arriv_times) / 2)):
            sta = sta_phase[i][:-2]
            time_P = arriv_times[i]
            time_S = arriv_times[i + int(len(arriv_times) / 2)]
            diff = abs(time_P-time_S)
            with open(output_file_path, "a", encoding="utf-8") as output_file:
                output_file.write(
                    f"{idx},{sta},{time_P},{time_S},{diff},"
                    f"{lon_event},{lat_event},{z_event}\n"
                )
else:
    print(f"The output file '{output_file_path}' already exists. No modifications are made.")

# ========== Doing Maths ==========

# Read the data from the file
with open(output_file_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    data = [line for line in reader]

for line in data:
    p_wave = float(line[2])
    s_wave = float(line[3])
    differences = float(line[4])

# Organize data for visualization
templates = list(set(line[0] for line in data))
stations = list(set(line[1] for line in data))
p_waves_times_template = {template: [] for template in templates}
p_waves_times_station = {station: [] for station in stations}
s_waves_times_template = {template: [] for template in templates}
s_waves_times_station = {station: [] for station in stations}

for line in data:
    template = line[0]
    station = line[1]
    p_waves_times_template[template].append(float(line[2]))
    p_waves_times_station[station].append(float(line[2]))
    s_waves_times_template[template].append(float(line[3]))
    s_waves_times_station[station].append(float(line[3]))

# ========== Plot Section ==========
counts=[]
# Organize data for visualization
templates = list(set(line[0] for line in data))
# templates = ['1564','1619','1747','1646','609','1801','1936','110','695','1627',,'2232','2233','2234','2235','2236','2237','2238'
#               '1939','1630','1807','1940','1804','1634','1945','1800','1817','1793','1642']'2226',
# templates = [str(num) for num in range(1162, 1174)]
templates = ['2065']
template_windows = {}
# Iterate over each family
for template in templates:
    # Choosing stations
    template_data = [line for line in data if line[0] == template and line[1] not in ['BPCB', 'TWGB', 'GOWB', 'LCBC', 'SOKB']]
    template_data = [line for line in data if line[0] == template and line[1] in ['MGCB', 'JRBC', 'SILB','SSIB', 'TSJB', 'KLNB']]

    # Extract station names and corresponding P and S wave times
    stations = [line[1] for line in template_data]
    p_wave_times = [float(line[2]) for line in template_data]
    s_wave_times = [float(line[3]) for line in template_data]

    # Sort stations by P-wave times in reverse order while keeping corresponding S-wave times aligned
    sorted_data = sorted(zip(p_wave_times, stations, s_wave_times), reverse=True)
    p_wave_times, stations, s_wave_times = zip(*sorted_data)

    # Create a new figure for each template
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot grey lines for each station
    for station in stations:
        ax.plot([0, 30], [station, station], color='grey', linestyle='-', linewidth=1)

    # Plot red dot for S wave time
    ax.scatter(s_wave_times, stations, color='red', marker='o', label='S Wave')

    # Plot blue dot for P wave time
    ax.scatter(p_wave_times, stations, color='blue', marker='o', label='P Wave')

    # Set x-axis ticks
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Station')
    ax.set_title(f'Wave Times for template {template}')

    # Display legend
    ax.legend(loc='upper right')

    # Draw a black line between the two dots for each station
    for s_wave, p_wave, station in zip(s_wave_times, p_wave_times, stations):
        ax.plot([p_wave, s_wave], [station, station], color='black', linestyle='-', linewidth=2)

    # Maths
    min_p_wave = round(min(p_wave_times), 3)
    max_p_wave = round(max(p_wave_times), 3)
    min_s_wave = round(min(s_wave_times), 3)
    max_s_wave = round(max(s_wave_times), 3)
    mean_p_wave = round(sum(p_wave_times) / len(p_wave_times), 3)
    mean_s_wave = round(sum(s_wave_times) / len(s_wave_times), 3)
    median_p_wave = round(np.median(p_wave_times), 3)
    median_s_wave = round(np.median(s_wave_times), 3)
    percentile_25_p_wave = round(np.percentile(p_wave_times, 25), 3)
    percentile_75_s_wave = round(np.percentile(s_wave_times, 75), 3)
    percentile_10_p_wave = round(np.percentile(p_wave_times, 10), 3)
    percentile_90_s_wave = round(np.percentile(s_wave_times, 90), 3)
    max_s_min_p_diff = round(max_s_wave - min_p_wave, 3)
    differences = [s - p for s, p in zip(s_wave_times, p_wave_times)]
    # mean_difference = round(sum(differences) / len(differences), 3)
    percentile_90_difference = round(np.percentile(differences, 90), 3)
    
    count_stations = 0
    # Define the interval (= template matching window)
    interval_lower = percentile_75_s_wave 
    interval_upper = percentile_75_s_wave - percentile_90_difference
    for i in range(len(stations)):
        p_time=p_wave_times[i]
        s_time=s_wave_times[i]
        if interval_lower <= p_time <= interval_upper and interval_lower <= s_time <= interval_upper:
            count_stations += 1

    # Add vertical lines on the x-axis
    ax.axvline(x=interval_lower, color='red', linestyle='--')
    ax.axvline(x=interval_upper, color='red', linestyle='--')

    # Store the information in a dictionary
    template_windows[template] = {
        'percentile_75_s_wave': percentile_75_s_wave,
        'min_p_wave': min_p_wave,
        'max_p_wave': max_p_wave,
        'min_s_wave': min_s_wave,
        'max_s_wave': max_s_wave,
        'max_s_min_p_diff': max_s_min_p_diff,
        'mean_p_wave': mean_p_wave,
        'mean_s_wave': mean_s_wave,
        'median_p_wave': median_p_wave,
        'median_s_wave': median_s_wave,
        'interval choosen': 'min P-wave arrival + win_size(=10)',
        'how many stas in interval': count_stations,
        'stations': stations
    }
    plt.show()
    print(template_windows)
    # print(count_stations)
    # counts.append(count_stations)
# bins = list(range(0, 21, 2))
# hist_values, bin_edges, _ = plt.hist(counts, bins=bins)
# plt.xlabel('Number of Stations')
# plt.ylabel('Frequency')
# plt.title('Histogram of Stations with P and S wave arrivals (90th)')
# plt.show()
np.save('windows_param_tim.npy', template_windows)

# test=np.load('windows_param.npy', allow_pickle=True).item()


# =============================================================================
#     # # Calculate the minimum P-wave time and minimum S-wave time
#     # min_p_wave_time = min(p_wave_times)
#     # min_s_wave_time = min(s_wave_times)
# 
#     # # Determine the time interval start based on the smaller of the two
#     # time_interval_start = min_p_wave_time if min_p_wave_time < min_s_wave_time else min_s_wave_time
# 
#     # # Determine the time interval end based on the larger of the two
#     # max_p_wave_time = max(p_wave_times)
#     # max_s_wave_time = max(s_wave_times)
#     # time_interval_end = max_p_wave_time if max_p_wave_time > max_s_wave_time else max_s_wave_time
# 
#     # Print the minimum P-wave and maximum S-wave times for each template
#     # print(f"Template {template}: Minimum P-wave time = {min_p_wave_time}, Maximum S-wave time = {max_s_wave_time}")
#     # time_difference = max_s_wave_time - min_p_wave_time
#     # import math
#     # print(f"Template {template}: Rounded Difference = {math.ceil(time_difference)}")
#     # # Add vertical lines at the determined time interval
#     # ax.axvline(x=time_interval_start, color='red', linestyle='--')
#     # ax.axvline(x=time_interval_end, color='red', linestyle='--')
# =============================================================================

    # Save figure
    # plt.savefig(base_dir + f'plots/tim/arrival times/wave_times_template_{template}.png')
    
# print(template_info)
# =============================================================================
# # Plotting Median Times for P-waves and S-waves for Each Family
# plt.figure(figsize=(10, 6))
# sorted_families_p = sorted(p_waves_times_template.keys())
# plt.bar(sorted_families_p, [np.median(p_waves_times_template[family]) for family in sorted_families_p])
# plt.title('Median Times for P-waves - Each Template')
# plt.xlabel('Family')
# plt.ylabel('Time (s)')
# plt.xticks(rotation=45, ha='right')
# for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
#     if i % 20 != 0:
#         label.set_visible(False)
# plt.tight_layout()
# plt.savefig('median_p_waves_template.png')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# sorted_families_s = sorted(s_waves_times_template.keys())
# plt.bar(sorted_families_s, [np.median(s_waves_times_template[family]) for family in sorted_families_s])
# plt.title('Median Times for S-waves - Each Template')
# plt.xlabel('Family')
# plt.ylabel('Time (s)')
# plt.xticks(rotation=45, ha='right')
# for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
#     if i % 20 != 0:
#         label.set_visible(False)
# plt.tight_layout()
# plt.savefig('median_s_waves_template.png')
# plt.show()
#
# # Plotting Median Times for P-waves and S-waves for Each Station
# plt.figure(figsize=(10, 6))
# station_keys_p = list(p_waves_times_station.keys())
# plt.bar(station_keys_p, [np.median(p_waves_times_station[station]) for station in station_keys_p])
# plt.title('Median Times for P-waves - Each Station')
# plt.xlabel('Station')
# plt.ylabel('Time (s)')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('median_p_waves_station.png')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# station_keys_s = list(s_waves_times_station.keys())
# plt.bar(station_keys_s, [np.median(s_waves_times_station[station]) for station in station_keys_s])
# plt.title('Median Times for S-waves - Each Station')
# plt.xlabel('Station')
# plt.ylabel('Time (s)')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('median_s_waves_station.png')
# plt.show()
#
# # Plotting Median Times for P and S Waves - Each Station (Sorted by P-wave Median)
# fig, ax = plt.subplots(figsize=(14, 8))
# bar_width = 0.35
# index = np.arange(len(stations))
# ax.bar(index, [np.median(p_waves_times_station[station]) for station in stations], bar_width, color='blue', alpha=0.7, label='Median P Wave')
# ax.bar(index + bar_width, [np.median(s_waves_times_station[station]) for station in stations], bar_width, color='red', alpha=0.7, label='Median S Wave')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(stations, rotation=45, ha='right')
# ax.set_xlabel('Station')
# ax.set_ylabel('Time (s)')
# ax.set_title('Median Times for P and S Waves - Each Station')
# ax.legend()
# plt.tight_layout()
# plt.savefig('median_p_and_s_waves_station.png')
# plt.show()
#
# # Calculate median P-wave and S-wave times for each station
# # median_p_waves = [np.median(p_waves_times_station[station]) for station in stations]
# # median_s_waves = [np.median(s_waves_times_station[station]) for station in stations]
# median_p_waves = [np.median(p_waves_times_template[template]) for template in templates]
# median_s_waves = [np.median(s_waves_times_template[template]) for template in templates]
#
# # Sort stations based on median P-wave times
# sorted_indices_p = np.argsort(median_p_waves)[::-1]  # Sort in descending order
# # sorted_stations_p = [stations[i] for i in sorted_indices_p]
# sorted_templates_p = [templates[i] for i in sorted_indices_p]
# sorted_median_p_waves = [median_p_waves[i] for i in sorted_indices_p]
#
# # Sort stations based on median S-wave times
# sorted_indices_s = np.argsort(median_s_waves)[::-1]  # Sort in descending order
# # sorted_stations_s = [stations[i] for i in sorted_indices_s]
# sorted_templates_s = [templates[i] for i in sorted_indices_p]
# sorted_median_s_waves = [median_s_waves[i] for i in sorted_indices_s]
#
# # Create a figure with a bar plot for each station
# fig, ax = plt.subplots(figsize=(14, 8))
#
# bar_width = 0.35
# index = np.arange(len(templates))
#
# # Plot bars for median P wave times (in a less bright blue with transparency)
# ax.bar(index, sorted_median_p_waves, bar_width, color='blue', alpha=0.7, label='Median P Wave')
#
# # Plot bars for median S wave times (in a less bright red with transparency)
# ax.bar(index + bar_width, sorted_median_s_waves, bar_width, color='red', alpha=0.7, label='Median S Wave')
#
# # Set x-axis ticks and labels
# ax.set_xticks(index + bar_width / 2)
# # ax.set_xticklabels(sorted_stations_p, rotation=45, ha='right')  # Use sorted stations for labeling
# ax.set_xticklabels(sorted_templates_p, rotation=45, ha='right')  # Use sorted stations for labeling
#
# # ax.axhline(y=13, color='black', linestyle='--')
# # ax.axhline(y=21, color='black', linestyle='--')
# ax.axhline(y=13, color='black', linestyle='--')
# ax.axhline(y=21, color='black', linestyle='--')
#
# # Set labels and title
# ax.set_xlabel('Station')
# ax.set_xlabel('Template')
# ax.set_ylabel('Time (s)')
# # ax.set_title('Median Times for P and S Waves - Each Station (Sorted by P-wave Median)')
# ax.set_title('Median Times for P and S Waves - Each Template (Sorted by P-wave Median)')
# ax.legend()
#
# # Save figure
# # plt.savefig('interval_station_plot_sorted.png')
# plt.savefig('interval_template_plot_sorted.png')
# plt.tight_layout()
# plt.show()
#
# =============================================================================
