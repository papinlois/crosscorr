# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:37 2023

@author: papin
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colorbar as clrbar
from scipy import stats
import autocorr_tools

def plot_station_locations(locs):
    """
    Create a plot of station locations on a map.

    Parameters:
        locs (list of tuples): A list of tuples, where each tuple contains
        station name, longitude, and latitude.

    This function creates a scatter plot of station locations and labels them
    with station names. The resulting plot is saved as 'station_locations.png'.

    Returns:
        None
    """
    plt.figure()
    plt.plot(locs[:, 1], locs[:, 2], 'bo')
    for name, lon, lat in locs:
        plt.text(lon, lat, name)
    plt.savefig('C:/Users/papin/Desktop/phd/plots/station_locations.png')
    plt.close()

def merge_csv_data(csv_file_paths, date_to_find, hour_of_interest=None):
    """
    Merge and process data from multiple CSV files.

    Parameters:
        csv_file_paths (list of str): List of file paths to the CSV data files.
        date_to_find (str): The date to filter by in the 'starttime' column
            (in the format "YYYY-MM-DD").
        hour_of_interest (int, optional): The specific hour to filter by (0-23).
            If not provided, data from all hours is included.

    This function extracts data from CSV files, filters it by date and, if
    specified, hour, and returns a processed DataFrame with unique 'starttime'
    values.

    Returns:
        df_no_duplicates (pd.DataFrame): The merged, filtered, and sorted
        DataFrame with duplicate rows removed. (only want starttime of events)
    """
    # Read CSV files into DataFrames
    data_frames = [pd.read_csv(file_path) for file_path in csv_file_paths]

    # Filter the DataFrames by date and, if specified, hour
    filtered_data_frames = []

    for df in data_frames:
        if hour_of_interest is not None:
            filter_condition = (df['starttime'].str.contains(date_to_find)) & (df['starttime'].str.contains(f"T{hour_of_interest:02d}:"))
            filtered_data_frames.append(df[filter_condition])
        else:
            filter_condition = df['starttime'].str.contains(date_to_find)
            filtered_data_frames.append(df[filter_condition])

    # Merge the filtered DataFrames
    merged_df = pd.concat(filtered_data_frames, ignore_index=True)

    # Sort by the "starttime" column
    sorted_df = merged_df.sort_values(by="starttime", ascending=True)

    # Remove duplicate rows based on the "starttime" column
    df_no_duplicates = sorted_df.drop_duplicates(subset="starttime")
    
    # Reset the index to start from 1
    df_no_duplicates.reset_index(drop=True, inplace=True)

    return df_no_duplicates

def create_detection_plot(aboves, xcorrmean, detection_plot_filename):
    """
    Create a detection plot.

    Parameters:
        aboves (tuple): A tuple containing two arrays - aboves[0] for Template Index
            and aboves[1] for Time Index.
        xcorrmean (numpy.ndarray): Array containing the correlation coefficients.
        detection_plot_filename (str): The file path to save the detection plot.
    """
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(aboves[0], aboves[1], s=20, c=xcorrmean[aboves])
    ax.set_xlabel('Template Index', fontsize=14)
    ax.set_ylabel('Time Index', fontsize=14)
    cax, _ = clrbar.make_axes(ax)
    cbar = clrbar.ColorbarBase(cax)
    cbar.ax.set_ylabel('Correlation Coefficient', rotation=270, 
                       labelpad=15, fontsize=14)
    ax.set_xlim((np.min(aboves[0]), np.max(aboves[0])))
    ax.set_ylim((np.min(aboves[1]), np.max(aboves[1])))
    plt.savefig(detection_plot_filename)
    plt.close()

def plot_cross_correlation(xcorrmean, aboves, thresh, mad, windowlen, st, hour_of_interest, date_of_interest, correlation_function_plot_filename):
    """
    Plot the cross-correlation function with template.

    Parameters:
        xcorrmean (numpy.ndarray): 2D array containing cross-correlation values.
        aboves (tuple): Tuple of arrays containing significant correlation indices.
        thresh (float): Threshold value.
        mad (float): Median absolute deviation.
        windowlen (int): Template window length in points.
        st (obspy.core.stream.Stream): ObsPy stream object containing the data.
        hour_of_interest (int): The specific hour of interest.
        date_of_interest (str): The date of interest (in the format "YYYYMMDD").
        correlation_function_plot_filename (str): Path to save the plot.

    Returns:
        None
    """
    winind = stats.mode(aboves[0], keepdims=False)[0]  # Most common value (template)
    xcorr = xcorrmean[winind, :]
    _, ax = plt.subplots(figsize=(10, 3))
    t = st[0].stats.delta * np.arange(len(xcorr))
    ax.plot(t, xcorr)
    ax.axhline(thresh * mad, color='red')
    inds = np.where(xcorr > thresh * mad)[0]
    clusters = autocorr_tools.clusterdects(inds, windowlen)
    newdect = autocorr_tools.culldects(inds, clusters, xcorr)
    ax.plot(newdect * st[0].stats.delta, xcorr[newdect], 'kx')
    ax.text(60, 1.1 * thresh * mad, '8*MAD', fontsize=16, color='red')
    ax.set_xlabel(f'Seconds of Hour {hour_of_interest} on {date_of_interest}', fontsize=14)
    ax.set_ylabel('Correlation Coefficient', fontsize=14)
    ax.set_xlim((0, 3600))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig(correlation_function_plot_filename)
    plt.close()

def plot_data(st, stas, channels):
    """
    Plot station data.

    Parameters:
        st (obspy.core.stream.Stream): Stream containing seismic data.
        stas (list of str): List of station names.
        channels (list of str): List of channel names.

    This function plots the seismic data from the specified stations and channels,
    normalizing each trace and adding an offset for visualization.

    Returns:
        cha (str): The channel name used for plotting.
    """
    plt.figure(figsize=(15, 5))
    offset = 0
    for sta_idx, sta in enumerate(stas):
        for cha_idx, cha in enumerate(channels):
            # Calculate color shade
            shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
            color = (0, 0, 0.5 + shade / 2)
            tr = st[sta_idx * len(channels) + cha_idx]
            plt.plot(tr.times("timestamp"), tr.data / np.max(np.abs(tr.data)) + offset,
                      color=color, label=f"{sta}_{cha}")
            offset += 1
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.savefig('C:/Users/papin/Desktop/phd/plots/data_plot.png')
    plt.close()
    
    return cha

def append_to_file(filename, thresh_mad, max_xcorr):
    """
    Append threshold and maximum cross-correlation values to a text file.
    """
    with open(filename, "a", encoding='utf-8') as file:
        file.write(f"{thresh_mad}\t{max_xcorr}\n")

def plot_scatter_from_file(file_path):
    """
    Read data from a file and create a scatter plot.

    Parameters:
        file_path (str): The path to the file containing the data.

    This function reads data from a file, assumes a specific format, and creates
    a scatter plot based on the read data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        thresh_mad_values = []
        max_xcorr_values = []
        for line in lines[1:]:  # Skip the header line
            parts = line.split('\t')
            thresh_mad_values.append(float(parts[0]))
            max_xcorr_values.append(float(parts[1]))

    x_values = range(1, len(thresh_mad_values) + 1)
    plt.scatter(x_values, thresh_mad_values, c='blue', label='Thresh * Mad')
    plt.scatter(x_values, max_xcorr_values, c='red', label='Max Xcorr')

    plt.xticks(range(1, len(x_values) + 1))
    plt.yticks([i * 0.1 for i in range(6)] + [1])
    max_y = math.ceil(max(max(thresh_mad_values), max(max_xcorr_values)) / 0.1) * 0.1 + 0.1
    plt.ylim(0, max_y)
    plt.grid(axis='x')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xlabel('Number')
    plt.ylabel('Values of Correlation')
    plt.legend()
    plt.show()

