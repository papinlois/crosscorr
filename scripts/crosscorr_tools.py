# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:37 2023

This module provides functions for seismic data processing, visualization, and analysis.
The functions cover loading and preprocessing data, plotting seismic traces, creating
summed traces around detected events, and more.

Functions:
- get_traces: Load seismic data for specified stations, channels, and date.
- get_traces_PB: Load seismic data from the PB network for specified stations,
  channels, and date.
- process_data: Preprocess seismic data, including interpolation, trimming,
  detrending, and filtering.
- plot_data: Plot seismic station data for a specific date, including normalized
  traces with an offset.
- plot_locations: Create a plot of station locations and template events on a map.
- plot_stack: Plot the combined traces for all detections of a template.

@author: papin

As of 11/27/23.
"""

import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import Stream, read

def get_traces(pairs, date_of_interest, base_dir):
    """
    Retrieve seismic traces for specified stations and channels on a given date.

    Parameters:
        stas (list): List of station names.
        channels (list): List of channel names.
        date_of_interest (str): The date in the format 'YYYYMMDD'
        base_dir (str): The base directory for file paths.

    Returns:
        - list: List of ObsPy Trace objects containing the seismic data.

    Note:
    - The function yields a Trace object for each station and channel combination.
    """
    path = os.path.join(base_dir, 'data', 'seed')
    for pair in pairs:
        file = os.path.join(path, f"{date_of_interest}.CN.{pair}.mseed")
        try:
            yield read(file)[0]
        except FileNotFoundError:
            print(f"File {file} not found.")

def process_data(st, sampling_rate, freqmin, freqmax,startdate, enddate):
    """
    Preprocess seismic data.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        sampling_rate (float): Sampling rate for interpolation.
        freqmin (float): Minimum frequency for bandpass filtering.
        freqmax (float): Maximum frequency for bandpass filtering.

    Returns:
        st (obspy.core.Stream): Seismic data streams.
    """
    # Initialize start and end with values that will be updated
    starttime = UTCDateTime(startdate)
    endtime = UTCDateTime(enddate)

    # Preprocessing: Interpolation, trimming, detrending, and filtering
    for tr in st:
        tr.trim(starttime=starttime, endtime=endtime, pad=1, fill_value=0)
        if tr.stats.sampling_rate != sampling_rate:
            tr.interpolate(sampling_rate=sampling_rate, starttime=starttime)
        # TODO: try with and without this and see if there is a difference?
        # tr.detrend(type='simple')
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

    return st

def plot_data(st, stas, channels, base_dir):
    """
    Plot seismic station data for a specific date.

    Parameters:
        for which you want to plot the data.
        stas (list): List of station names.
        channels (list): List of channel names.
        base_dir (str): The base directory for file paths.

    Returns:
        None
    """
    plt.figure(figsize=(15, 7))
    nb = 15 # Distance between plots
    offset = len(stas) * len(channels) * nb

    # Get the start date from the first trace in the stream
    start_date = st[0].stats.starttime.strftime("%Y%m%d")

    # Plot the data
    for sta_idx, sta in enumerate(stas):
        for cha_idx, cha in enumerate(channels):
            # Calculate color shade
            shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
            color = (0, 0.2, 0.5 + shade / 2)
            tr = st[sta_idx * len(channels) + cha_idx]
            time_in_seconds = np.arange(len(tr.data)) * tr.stats.delta
            norm = np.median(3 * np.abs(tr.data))
            plt.plot(time_in_seconds, tr.data / norm + offset,
                      color=color, label=f"{sta}..{cha}")
            offset -= nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Full day of {start_date}', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.xlim(0, max(time_in_seconds))
    plt.ylim(0, len(stas) * len(channels) * nb + 10)
    plt.savefig(os.path.join(base_dir, 'plots', f'data_plot_{start_date}.png'))
    plt.show()

def plot_stack(utc_times, cc_values, tr1, win_size, template, template_index,
               iid, date_of_interest, base_dir):
    """
    Plot the combined traces for a detection and its corresponding template.

    Parameters:
        utc_times (list): All detections for tr1 and template.
        cc_values (numpy.ndarray): Array of cross-correlation values.
        tr1 (obspy.core.Trace): Seismic data trace.
        win_size (int): Size of the time window.
        template (obspy.core.Trace): Seismic data trace of the template.
        template_index (int): Index of the template.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
        base_dir (str): The base directory for file paths.

   Returns:
        None
    """
    crosscorr_combination = f'{iid}_templ{template_index}'

    # Stack of detection for 1 template and 1 sta..cha
    summed_traces=[]
    for i, utc_time in enumerate(utc_times):
        # Define the time window
        start_time = UTCDateTime(utc_time)
        start_window = start_time
        end_window = start_time + win_size
        # Extract the traces within the time window
        windowed_traces = tr1.slice(starttime=start_window, endtime=end_window)
        # # Sum the traces within the time window
        # summed_traces.append(windowed_traces.data)

        # TODO: Normalization to validate by amt
        max_abs_value = np.max(np.abs(windowed_traces.data))
        summed_traces.append((windowed_traces.data / max_abs_value) * cc_values[i])
    stack=np.sum(summed_traces,axis=0)

    ## 1st figure : detections stack
    plt.figure(figsize=(10, 4))
    num_pts_stack = len(stack)
    times_detection = np.linspace(0, win_size, num_pts_stack, endpoint=False)
    plt.plot(times_detection, stack,
             label=f'Stack of {len(utc_times)} detections', linestyle='-', color='C0')

    # Label the x-axis, y-axis, and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Stack of {len(utc_times)} detections {crosscorr_combination} - {date_of_interest}')
    plt.xlim(0, win_size)
    plt.grid(True)

    # Save the plot as an image file
    save_path = os.path.join(base_dir, 'plots',
                              f'templ{template_index}_stack_{iid}_{date_of_interest}.png')
    plt.savefig(save_path)
    plt.close()

    # ## 2nd figure : detections stack and template
    # _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # # Plot the summed trace for the detection
    # axs[0].plot(times_detection, stack,
    #             label=f'Stack of {len(utc_times)} detections', linestyle='-', color='C0')
    # axs[0].set_ylabel('Amplitude')
    # axs[0].grid(True)

    # # Plot the summed trace for the template
    # num_pts_template = len(template)
    # times_template = np.linspace(0, win_size, num_pts_template, endpoint=False)
    # axs[1].plot(times_template, template, 
    #             label=f'Template {template_index}', linestyle='--', color='C1')
    # axs[1].set_ylabel('Amplitude')
    # axs[1].grid(True)

    # # Add legend for the entire figure
    # axs[0].legend(loc='upper right')
    # axs[1].legend(loc='upper right')

    # plt.tight_layout()

    # # Save the plot as an image file
    # save_path = os.path.join(base_dir, 'plots',
    #                            f'templ{template_index}_stack_vs_{iid}_{date_of_interest}.png')
    # plt.savefig(save_path)
    # plt.close()

def plot_locations(locs, base_dir, events=None):
    """
    Create a plot of station locations and template events on a map.

    Parameters:
        locs (list): A list of tuples, where each tuple contains station name,
        longitude, and latitude.
        base_dir (str): The base directory for file paths.
        events (DataFrame or None): A DataFrame containing event data with
        columns 'lon','lat','depth', and 'datetime', or None if no events are provided.

    This function creates a scatter plot of station locations and labels them
    with station names. If events are provided, the function also plots the
    events and labels them with 'Events'. The resulting plot is saved as
    'station_events_locations_{date}.png'.

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

    # Plot events if provided
    if events is not None:
        plt.scatter(events['lon'], events['lat'], c='grey', marker='x', label='Events')
        # plt.colorbar(scatter, label='Depth') #c='events['depth']'
        date=(UTCDateTime(events['datetime'][0]).date).strftime('%Y%m%d')

    plt.legend(loc='upper right')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Station and Events Locations on {date}')
    plt.savefig(os.path.join(base_dir, 'plots', f'station_events_locations_{date}.png'))
    plt.close()

def get_traces_PB(stas, channels, startdate, network, base_dir): #update needed
    """
    Retrieve seismic traces for specified stations and channels on a given date,
    for the network PB.

    Parameters:
        stas (list): List of station names.
        channels (list): List of channel names.
        date_of_interest (str): The date in the format 'YYYYMMDD'
        base_dir (str): The base directory for file paths.

    Returns:
        - list: List of ObsPy Trace objects containing the seismic data.

    Note:
    - The function yields a Trace object for each station and channel combination.
    """
    day_of_year = startdate.timetuple().tm_yday
    year = startdate.timetuple().tm_year
    path = os.path.join(base_dir, 'data', 'seed')
    for sta in stas:
        file = os.path.join(path, f"{sta}.{network}.{year}.{day_of_year}")
        try:
            for cha in range(len(channels)):
                yield read(file)[cha]
        except FileNotFoundError:
            print(f"File {file} not found.")