#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:09:14 2024

This script creates a .txt file that relates estimated times for P- and 
S-waves for each of Bostock LFE families + some figures to show the data.

@author: lpapin
"""

import os
import numpy as np

import csv
import matplotlib.pyplot as plt

'''
base_dir = "/Users/lpapin/Documents/phd/"

def idx_loc(loc):
    """
    Given a rough location, return the index of the closest loc in grid-nodes.
    """
    d2 = np.sum((coords[:, :3] - loc)**2, axis=1)
    idx = np.argmin(d2)
    return idx

# Load data
# A = np.load('./all_T_0.1_3.npy', allow_pickle=True).item()
TT = np.load('Travel.npy', allow_pickle=True).item()
sav_family_phases = np.load('./sav_family_phases.npy', allow_pickle=True).item()

# Extract necessary data
coords = np.array(list(TT['T'].keys()))
sta_phase = TT['sta_phase']
family_nb = np.array(list(sav_family_phases.keys()))
output_file_path = os.path.join(base_dir, 'arrival_times_bostock.txt')

# Check if the output file already exists
if not os.path.isfile(output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Write header if the file is empty
        output_file.write("lfe_family,sta,P-wave,S-wave\n")#",lon_family,lat_family,z_family\n")

    # Iterate through LFE families
    for nb in family_nb:
        eqLoc_values = sav_family_phases[nb]['eqLoc']
        lon_family, lat_family, z_family = eqLoc_values
        exact_loc = coords[idx_loc([-lon_family, lat_family, -z_family])]
        arriv_times = TT['T'][tuple(exact_loc)]

        # Write data for each station and its P- and S-wave arrival times
        for i in range(0, int(len(arriv_times) / 2)):
            sta = sta_phase[i][:-2]
            time_P = arriv_times[i]
            time_S = arriv_times[i + int(len(arriv_times) / 2)]
            with open(output_file_path, "a", encoding="utf-8") as output_file:
                output_file.write(
                    f"{nb},{sta},{time_P},{time_S}\n"
                    # f"{lon_family},{lat_family},{z_family}\n"
                )
else:
    print(f"The output file '{output_file_path}' already exists. No modifications are made.")
'''

# Some maths for our sake

# Read the data from the file
with open('arrival_times_bostock.txt', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    data = [line for line in reader]

# Calculate differences
differences = []
for line in data:
    p_wave = float(line[2])
    s_wave = float(line[3])
    differences.append(s_wave - p_wave)

# Organize data for visualization
families = list(set(line[0] for line in data))
stations = list(set(line[1] for line in data))
p_waves_times_family = {family: [] for family in families}
p_waves_times_station = {station: [] for station in stations}
s_waves_times_family = {family: [] for family in families}
s_waves_times_station = {station: [] for station in stations}

for line in data:
    family = line[0]
    station = line[1]
    p_waves_times_family[family].append(float(line[2]))
    p_waves_times_station[station].append(float(line[2]))
    s_waves_times_family[family].append(float(line[3]))
    s_waves_times_station[station].append(float(line[3]))

## Plotting
'''
# Plotting Median Times for P-waves and S-waves for Each Family
plt.figure(figsize=(10, 6))
sorted_families_p = sorted(p_waves_times_family.keys())
plt.bar(sorted_families_p, [np.median(p_waves_times_family[family]) for family in sorted_families_p])
plt.title('Median Times for P-waves - Each Family')
plt.xlabel('Family')
plt.ylabel('Time (s)')
plt.xticks(rotation=45, ha='right')
for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
    if i % 20 != 0:
        label.set_visible(False)
plt.tight_layout()
plt.savefig('median_p_waves_family.png')
plt.show()

plt.figure(figsize=(10, 6))
sorted_families_s = sorted(s_waves_times_family.keys())
plt.bar(sorted_families_s, [np.median(s_waves_times_family[family]) for family in sorted_families_s])
plt.title('Median Times for S-waves - Each Family')
plt.xlabel('Family')
plt.ylabel('Time (s)')
plt.xticks(rotation=45, ha='right')
for i, label in enumerate(plt.gca().xaxis.get_ticklabels()):
    if i % 20 != 0:
        label.set_visible(False)
plt.tight_layout()
plt.savefig('median_s_waves_family.png')
plt.show()

# Plotting Median Times for P-waves and S-waves for Each Station
plt.figure(figsize=(10, 6))
station_keys_p = list(p_waves_times_station.keys())
plt.bar(station_keys_p, [np.median(p_waves_times_station[station]) for station in station_keys_p])
plt.title('Median Times for P-waves - Each Station')
plt.xlabel('Station')
plt.ylabel('Time (s)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('median_p_waves_station.png')
plt.show()

plt.figure(figsize=(10, 6))
station_keys_s = list(s_waves_times_station.keys())
plt.bar(station_keys_s, [np.median(s_waves_times_station[station]) for station in station_keys_s])
plt.title('Median Times for S-waves - Each Station')
plt.xlabel('Station')
plt.ylabel('Time (s)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('median_s_waves_station.png')
plt.show()

# Plotting Median Times for P and S Waves - Each Station (Sorted by P-wave Median)
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.35
index = np.arange(len(stations))
ax.bar(index, [np.median(p_waves_times_station[station]) for station in stations], bar_width, color='blue', alpha=0.7, label='Median P Wave')
ax.bar(index + bar_width, [np.median(s_waves_times_station[station]) for station in stations], bar_width, color='red', alpha=0.7, label='Median S Wave')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(stations, rotation=45, ha='right')
ax.set_xlabel('Station')
ax.set_ylabel('Time (s)')
ax.set_title('Median Times for P and S Waves - Each Station')
ax.legend()
plt.tight_layout()
plt.savefig('median_p_and_s_waves_station.png')
plt.show()
'''
'''
# Calculate median P-wave and S-wave times for each station
median_p_waves = [np.median(p_waves_times_station[station]) for station in stations]
median_s_waves = [np.median(s_waves_times_station[station]) for station in stations]

# Sort stations based on median P-wave times
sorted_indices_p = np.argsort(median_p_waves)[::-1]  # Sort in descending order
sorted_stations_p = [stations[i] for i in sorted_indices_p]
sorted_median_p_waves = [median_p_waves[i] for i in sorted_indices_p]

# Sort stations based on median S-wave times
sorted_indices_s = np.argsort(median_s_waves)[::-1]  # Sort in descending order
sorted_stations_s = [stations[i] for i in sorted_indices_s]
sorted_median_s_waves = [median_s_waves[i] for i in sorted_indices_s]

# Create a figure with a bar plot for each station
fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.35
index = np.arange(len(stations))

# Plot bars for median P wave times (in a less bright blue with transparency)
ax.bar(index, sorted_median_p_waves, bar_width, color='blue', alpha=0.7, label='Median P Wave')

# Plot bars for median S wave times (in a less bright red with transparency)
ax.bar(index + bar_width, sorted_median_s_waves, bar_width, color='red', alpha=0.7, label='Median S Wave')

# Set x-axis ticks and labels
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(sorted_stations_p, rotation=45, ha='right')  # Use sorted stations for labeling

ax.axhline(y=8, color='black', linestyle='--')
ax.axhline(y=16, color='black', linestyle='--')

# Set labels and title
ax.set_xlabel('Station')
ax.set_ylabel('Time (s)')
ax.set_title('Median Times for P and S Waves - Each Station (Sorted by P-wave Median)')
ax.legend()

# Save figure
# plt.savefig('interval_station_plot_sorted.png')
plt.tight_layout()
plt.show()
'''

# Plot 5

# Organize data for visualization
families = list(set(line[0] for line in data))

# Iterate over each family
for family in families:
    # if family=='001':
    # Filter data for the current family
    family_data = [line for line in data if line[0] == family]
    
    # Anchor stations but KELB : LZB,PGC,PFB,TWBB,MGCB,TSJB,TWKB
    family_data = [family_data[9],family_data[17],family_data[0],family_data[-2],family_data[16],family_data[5],family_data[1]]
    
    # Extract station names and corresponding P and S wave times
    stations = [line[1] for line in family_data]
    p_wave_times = [float(line[2]) for line in family_data]
    s_wave_times = [float(line[3]) for line in family_data]
    
    # Create a new figure for each family
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
    ax.set_title(f'Wave Times for Family {family}')
    
    # Display legend
    ax.legend()
    
    # Draw a black line between the two dots for each station
    for s_wave, p_wave, station in zip(s_wave_times, p_wave_times, stations):
        ax.plot([p_wave, s_wave], [station, station], color='black', linestyle='-', linewidth=2)
    
    # Add vertical lines at 8 and 16 on the x-axis
    ax.axvline(x=8, color='red', linestyle='--')
    ax.axvline(x=16, color='red', linestyle='--')

    # Save figure
    plt.savefig(f'wave_times_family_{family}.png')
    plt.close()
