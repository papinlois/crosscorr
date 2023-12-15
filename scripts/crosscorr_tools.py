# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:37 2023

This module provides functions for seismic data processing, visualization, and analysis.
The functions cover loading and preprocessing data, plotting seismic traces, creating
summed traces around detected events, and more.

Functions:
- build_file_path: Gives file names for plots.
- get_traces: Load seismic data for specified stations, channels, and date.
- process_data: Preprocess seismic data, including interpolation, trimming,
  detrending, and filtering.
- plot_data: Plot seismic station data for a specific date, including normalized
  traces with an offset.
- plot_crosscorr: Plot the resulted cross-correlation function for 1 template
  on the network.
- plot_template: Plot the template for each station/channel combination.
- plot_stacks: Plot the combined traces for all detections of a template for each
  station/channel combination.
- plot_locations: Create a plot of station locations and template events on a map.

@author: papin

As of 12/15/23.
"""

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import read, Stream

def build_file_path(base_dir, folder, name, prefix, lastday):
    """
    Build a file path for saving plots (related to plot_template, plot_stacks,
    plot_crosscorr in crosscorr_tools module).

    Parameters:
        base_dir (str): The base directory where the file will be saved.
        folder (str): The subfolder within the base directory for organization.
        name (str): The base name of the file (usually number of the template).
        prefix (str): A prefix to be included in the file name.
        lastday (str): The last day identifier to be included in the file name.
    """
    return os.path.join(base_dir, f'plots/{folder}/{name}_{prefix}_{lastday}.png')

def get_traces(network_config, date_of_interests, base_dir):
    """
    Retrieve seismic traces for specified stations and channels on a given dates.

    Parameters:
        network_config (dict): A dictionary containing configuration details for each network.
        date_of_interests (str): The dates in the format 'YYYYMMDD'
        base_dir (str): The base directory for file paths.

    Returns:
        st (obspy.core.Stream): Seismic data streams.
    """
    path = os.path.join(base_dir, 'data', 'seed')

    st = Stream()

    for date in date_of_interests:
        for network, config in network_config.items():
            stations = config['stations']
            channels = config['channels']
            filename_pattern = config['filename_pattern']
            for sta in stations:
                if network == 'PB':
                    dt = datetime.strptime(date, '%Y%m%d')
                    julian_day = (dt - datetime(dt.year, 1, 1)).days + 1
                    file = os.path.join(path,
                                        filename_pattern.format(station=sta, year=dt.year, julian_day=julian_day))
                    try:
                        for i in range(3):
                            st += read(file)[i]
                    except FileNotFoundError:
                        print(f"File {file} not found.")
                else:
                    for cha in channels:
                        file = os.path.join(path,
                                            filename_pattern.format(date=date, station=sta, channel=cha))
                        try:
                            st += read(file)[0]
                        except FileNotFoundError:
                            print(f"File {file} not found.")

    # Merge all data by stations and channels
    st._trim_common_channels()
    st._cleanup()

    return st

def process_data(st, sampling_rate, freqmin, freqmax, startdate, enddate):
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
        tr.trim(starttime=starttime, endtime=endtime, pad=1,
                fill_value=0, nearest_sample=False)
        if tr.stats.sampling_rate != 100.0:
            tr.interpolate(sampling_rate=100.0, starttime=starttime, endtime=endtime)
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        tr.interpolate(sampling_rate=sampling_rate)

    return st

def plot_data(st, stas, channels, data_plot_filename):
    """
    Plot seismic station data for a specific date.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        stas (list): List of station names.
        channels (list): List of channel names.
        data_plot_filename (str): Filename for saving the plot.

    Returns:
        None
    """
    # Generate pairs based on stas and channels
    pairs = [f"{sta}..{cha}" for sta_list, cha_list in zip(stas, channels)
             for sta in sta_list for cha in cha_list]

    plt.figure(figsize=(16, 8))
    nb = 10  # Distance between plots
    offset = len(pairs) * nb

    # Get the start date from the first trace in the stream
    start_date = st[0].stats.starttime.strftime("%Y%m%d")

    # Plot the data based on the order specified in pairs
    for pair in pairs:
        sta, cha = pair.split('..')
        sta_idx = stas.index([s for s in stas if sta in s][0])
        cha_idx = channels[sta_idx].index(cha)
        tr = st[sta_idx * len(channels) + cha_idx]
        time_in_seconds = np.arange(len(tr.data)) * tr.stats.delta
        norm = np.median(3 * np.abs(tr.data))
        plt.plot(time_in_seconds, tr.data / norm + offset, label=f"{sta}..{cha}")
        offset -= nb

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Full day of {start_date}', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.xlim(0, max(time_in_seconds))
    plt.ylim(0, len(pairs) * nb + 10)
    plt.tight_layout()
    plt.savefig(data_plot_filename)
    plt.close()

def plot_crosscorr(st, xcorrmean, thresh, newdect, max_index,
                   name, lastday, crosscorr_plot_filename):
    """
    Plots cross-correlation data and saves the plot to a file.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        xcorrmean (numpy.ndarray): The cross-correlation mean values.
        thresh (float): Threshold for the new detections.
        newdect (numpy.ndarray): Indices of the detected events.
        max_index (int): Index of the maximum value in the cross-correlation.
        name (str): Combination identifier for the plot title.
        lastday (str): Date identifier for the plot title.
        crosscorr_plot_filename (str): Path to save the plot.

    Returns:
        None
    """
    tr=st[0]
    stream_duration = (tr.stats.endtime - tr.stats.starttime)
    _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(tr.stats.delta * np.arange(len(xcorrmean)), xcorrmean)
    ax.axhline(thresh, color='red')
    ax.plot(newdect * tr.stats.delta, xcorrmean[newdect], 'kx')
    ax.plot((newdect * tr.stats.delta)[max_index],
            (xcorrmean[newdect])[max_index], 'gx', markersize=10, linewidth=10)
    ax.text(60, 1.1 * thresh, '8*MAD', fontsize=14, color='red')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Correlation Coefficient', fontsize=14)
    ax.set_xlim(0, stream_duration)
    ax.set_title(f'{name}_{lastday}', fontsize=16)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(crosscorr_plot_filename)
    plt.close()

def plot_template(st, all_template, pairs, templ_idx, template_plot_filename):
    """
    Plot templates with an offset on the y-axis.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        all_template (list): List of template traces.
        pairs (list): List of pairs corresponding to each template in all_template.
        templ_idx (int): Index of the template.
        template_plot_filename (str): Filename for saving the plot.

    Returns:
        None
    """
    plt.figure(figsize=(12,6))
    nb = 1  # Distance between plots
    offset = len(all_template) * nb
    x = np.linspace(0, len(all_template[0]) / st[0].stats.sampling_rate,
                    len(all_template[0]), endpoint=False)
    
    # Plot each template with an offset on the y-axis
    for _, (template, pair) in enumerate(zip(all_template, pairs)):
        norm = np.max(np.abs(template.data))
        plt.plot(x, template / norm + offset, label=f'{pair}')
        offset -= nb

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'All Templates for Template {templ_idx}', fontsize=16)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.xlim(0, max(x))
    plt.ylim(0, len(pairs) * nb + nb)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(template_plot_filename)
    plt.close()

def plot_stacks(st, template, newdect, pairs, templ_idx, stack_plot_filename):
    """
    Plot the combined traces for a detection and its corresponding template.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        template (obspy.core.Trace): Seismic data trace of the template.
        newdect (numpy.ndarray): Indices of the detected events.
        pairs (list): List of pairs corresponding to each trace in st.
        templ_idx (int): Index of the template.
        stack_plot_filename (str): Filename for saving the plot.

    Returns:
        None
    """
    stacked_traces = np.zeros((len(st), len(template)))
    for idx, tr in enumerate(st):
        for dect in newdect:
            # TODO: do we normalize each detections before stacking?
            # max_abs_value = np.max(np.abs(tr.data[dect:dect + len(template)]))
            stacked_traces[idx, :] += tr.data[dect:dect + len(template)] #/ max_abs_value
    stacked_traces /= len(newdect)

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
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.xlim(0, max(x))
    plt.ylim(0, len(pairs) * nb + nb)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(stack_plot_filename)
    plt.close()

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
