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
import autocorr_tools

def plot_crosscorrelation(xcorrmean, thresh, mad, st, stream_duration, 
                          crosscorr_combination, date_of_interest):
    """
    Create a cross-correlation plot for a given event and station combination.

    Parameters:
        xcorrmean (numpy array): The cross-correlation function.
        thresh (float): The threshold value for detecting significant correlations.
        mad (float): The median absolute deviation value.
        st (obspy.core.Stream): The seismic data for the station.
        stream_duration (float): The duration of the seismic data in seconds.
        crosscorr_combination (str): A combination of station and event identifier.
        date_of_interest (str): The date of interest.

    This function creates a cross-correlation plot to visualize significant 
    correlations and their properties.

    Returns:
        None
    """
    # Calculate the window length for clustering
    windowlen = st[0].stats.npts
    
    # Find indices where the cross-correlation values are above the threshold
    inds = np.where(xcorrmean > thresh * mad)[0]
    
    # Cluster the detected events
    clusters = autocorr_tools.clusterdects(inds, windowlen)
    
    # Cull detections within clusters
    newdect = autocorr_tools.culldects(inds, clusters, xcorrmean)
    
    # Find the index of the maximum value in newdect
    max_index = np.argmax(xcorrmean[newdect])
    
    # Define the filename for the correlation plot
    correlation_plot_filename = (
        f'C:/Users/papin/Desktop/phd/plots/'
        f'crosscorr_{crosscorr_combination}_{date_of_interest}.png'
    )
    
    # Creation of the cross-correlation plot only if new events detected
    if newdect.size > 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        t = st[0].stats.delta * np.arange(len(xcorrmean))
        ax.plot(t, xcorrmean)
        ax.axhline(thresh * mad, color='red')
        ax.plot(newdect * st[0].stats.delta, xcorrmean[newdect], 'kx')
        ax.plot((newdect * st[0].stats.delta)[max_index], 
                (xcorrmean[newdect])[max_index], 'gx', markersize=10, linewidth=10)
        ax.text(60, 1.1 * thresh * mad, '8*MAD', fontsize=14, color='red')
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Correlation Coefficient', fontsize=14)
        ax.set_xlim(0, stream_duration)
        ax.set_title(f'{crosscorr_combination} - {date_of_interest}', fontsize=16)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.savefig(correlation_plot_filename)
        plt.close()

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
    
    # Separate stations by network and plot them in different colors
    for name, lon, lat, network in locs:
        if network == 'CN':
            plt.plot(lon, lat, 'bo')
        elif network == 'PB':
            plt.plot(lon, lat, 'ro')
        plt.text(lon, lat, name)
        
    plt.savefig('C:/Users/papin/Desktop/phd/plots/station_locations.png')
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
    nb = 10 # Distance between plots
    
    # Get the start date from the first trace in the stream
    start_date = st[0].stats.starttime.strftime("%Y%m%d")
    
    for sta_idx, sta in enumerate(stas):
        for cha_idx, cha in enumerate(channels):
            # Calculate color shade
            shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
            color = (0, 0, 0.5 + shade / 2)
            tr = st[sta_idx * len(channels) + cha_idx]
            time_in_seconds = np.arange(len(tr.data)) * tr.stats.delta
            norm=np.median(3*np.abs(tr.data))
            plt.plot(time_in_seconds, tr.data / norm + offset,
                      color=color, label=f"{sta}_{cha}")
            offset += nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Full day of {start_date}', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.ylim(-nb,nb*len(stas))
    plt.savefig(f'C:/Users/papin/Desktop/phd/plots/data_plot_{start_date}.png')
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
    # plt.xticks(range(1, len(x_values) + 1))
    plt.yticks([i * 0.1 for i in range(6)] + [1])
    max_y = math.ceil(max(max(thresh_mad_values), max(max_xcorr_values)) / 0.1) * 0.1 + 0.1
    plt.ylim(0, max_y)
    plt.grid(axis='x')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xlabel('Number')
    plt.ylabel('Values of Correlation')
    plt.legend()
    plt.show()

##############################################################################

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


