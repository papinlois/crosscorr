# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:37 2023

This module provides functions for seismic data processing, visualization, and analysis.
The functions cover loading and preprocessing data, plotting seismic traces, creating
summed traces around detected events, and more.

Functions:
- get_traces: Load seismic data for specified stations, channels, and date.
- get_traces_PB: Load seismic data from the PB network for specified stations, channels, and date.
- process_data: Preprocess seismic data, including interpolation, trimming, detrending, and filtering.
- plot_data: Plot seismic station data for a specific date, including normalized traces with an offset.
- plot_locations: Create a plot of station locations and template events on a map.
- plot_summed_traces: Preprocess seismic data and plot summed traces around detected events.
- plot_summed_traces_PB: Preprocess PB network seismic data and plot summed traces around detected events.
- plot_combined_traces: Plot the combined traces for a detection and its corresponding template

@author: papin
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import Stream, read

def get_traces(stas, channels, date_of_interest, base_dir):
    path = os.path.join(base_dir, 'data', 'seed')
    for cha in channels:
        for sta in stas:
            file = os.path.join(path, f"{date_of_interest}.CN.{sta}..{cha}.mseed")
            try:
                yield read(file)[0]
            except FileNotFoundError:
                print(f"File {file} not found.")

def get_traces_PB(stas, channels, startdate, network, base_dir):
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

def process_data(st, stas, locs=None, sampling_rate=80, freqmin=2.0, freqmax=8.0):
    """
    Preprocess seismic data.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        stas (list): List of station names.
        locs (numpy.ndarray, optional): Array containing station locations.
        sampling_rate (float): Sampling rate for interpolation.
        freqmin (float): Minimum frequency for bandpass filtering.
        freqmax (float): Maximum frequency for bandpass filtering.

    Returns:
        st (obspy.core.Stream): Seismic data streams.
        stas (list): List of station names.
    """
    # Filter out traces that don't have the required number of data points (a day)
    st = Stream(traces=[tr for tr in st if tr.stats.npts * tr.stats.delta == 86400])

    if not st:
        print("No traces have the required number of data points. Exiting process_data.")
        return

    # Preprocessing: Interpolation, trimming, detrending, and filtering
    for tr in st:
        tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.endtime, fill_value=0)
        tr.interpolate(sampling_rate=sampling_rate, starttime=tr.stats.starttime)
        tr.detrend(type='simple')
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

    # Filter stas based on available traces
    stas_with_data = [tr.stats.station for tr in st]
    stas = [sta for sta in stas if sta in stas_with_data]

    # Add locations if locs is provided
    if locs is not None:
        for sta_idx, sta in enumerate(stas):
            ind = np.where(locs[:, 0] == sta)
            st[sta_idx].stats.y = locs[ind, 1][0][0]
            st[sta_idx].stats.x = locs[ind, 2][0][0]

    return st, stas

def plot_data(date_of_interest, stas, channels, network, base_dir):
    """
    Plot seismic station data for a specific date.

    Parameters:
        date_of_interest (str): The date in the format 'YYYYMMDD'
        for which you want to plot the data.
        stas (list of str): List of station names.
        channels (list of str): List of channel names.
        network (str): Network code ('CN' or 'PB' for now).
        base_dir (str): The base directory for file paths.

    Returns:
        None
    """
    st = Stream()
    path = os.path.join(base_dir, 'data', 'seed')

    for sta in stas:
        if network == 'CN':
            for cha in channels:
                file = os.path.join(path, f"{date_of_interest}.CN.{sta}..{cha}.mseed")
                try:
                    st += read(file)[0]
                except FileNotFoundError:
                    print(f"File {file} not found.")
        elif network == 'PB':
            startdate = datetime.strptime(date_of_interest, "%Y%m%d")
            day_of_year = startdate.timetuple().tm_yday
            year = startdate.timetuple().tm_year
            file = os.path.join(path, f"{sta}.{network}.{year}.{day_of_year}")
            try:
                for cha in range(len(channels)):
                    st += read(file)[cha]
            except FileNotFoundError:
                print(f"File {file} not found.")

    st, stas = process_data(st, stas, locs=None, sampling_rate=80, freqmin=2.0, freqmax=8.0)

    plt.figure(figsize=(15, 5))
    nb = 10 # Distance between plots
    offset = len(stas) * len(channels) * nb

    # Get the start date from the first trace in the stream
    start_date = st[0].stats.starttime.strftime("%Y%m%d")

    # Plot the data
    for sta_idx, sta in enumerate(stas):
        for cha_idx, cha in enumerate(channels):
            # Calculate color shade
            shade = (sta_idx * len(channels) + cha_idx) / (len(stas) * len(channels))
            color = (0, 0.15, 0.5 + shade / 2)
            tr = st[sta_idx * len(channels) + cha_idx]
            time_in_seconds = np.arange(len(tr.data)) * tr.stats.delta
            norm = np.median(3 * np.abs(tr.data))
            plt.plot(time_in_seconds, tr.data / norm + offset,
                      color=color, label=f"{sta}_{cha}")
            offset -= nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Full day of {start_date}', fontsize=16)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True)
    plt.ylim(0, len(stas) * len(channels) * nb + 10)
    plt.savefig(os.path.join(base_dir, 'plots', f'data_plot_{start_date}.png'))
    plt.close()

def plot_locations(locs, base_dir, events=None):
    """
    Create a plot of station locations and template events on a map.

    Parameters:
        locs (list of tuples): A list of tuples, where each tuple contains
        station name, longitude, and latitude.
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

def plot_summed_traces(stas, channels, window_size, network, date_of_interest, base_dir, templates=None):
    """
    Preprocess seismic data and plot summed traces around detected events.

    Parameters:
        stas (list): List of station names.
        channels (list): List of channel names.
        window_size (int): Time window size in seconds.
        network (str): Network identifier.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
        base_dir (str): The base directory for file paths.
        templates (pd.DataFrame, optional): DataFrame containing template information.
            Defaults to None.

    Returns:
        None
    """
    # Get the data (so it can be called outside of crosscorr.py)
    st = Stream(traces=get_traces(stas, channels, date_of_interest, base_dir))
    channel_prefix = channels[0][:2]

    st, stas = process_data(st, stas, locs=None, sampling_rate=80, freqmin=2.0, freqmax=8.0)

    # Define the path to the detections file based on the provided date_of_interest
    output_file_path = os.path.join(base_dir, 'plots', f"{network} {channel_prefix} {date_of_interest}", 'output.txt')
    detections_df = pd.read_csv(output_file_path)

    # Create a dictionary to store summed traces for each template
    detect_summed_traces = {}

    # Iterate through the detections_df
    for _, detection in detections_df.iterrows():

        # Get the template index for the detection
        template_index = detection['templ']

        # Define the time window
        start_time = UTCDateTime(detection['starttime'])
        start_window = start_time
        end_window = start_time + window_size

        # Extract the traces within the time window
        windowed_traces = st.slice(starttime=start_window, endtime=end_window)

        # Manually sum the traces within the time window
        summed_traces = np.sum([trace.data for trace in windowed_traces], axis=0)

        # Append the summed trace to the template's list
        detect_summed_traces.setdefault(template_index, []).append(summed_traces)

    # Sum the traces for each template and plot
    for template_index, detect_traces in detect_summed_traces.items():
        # Sum the traces for the template
        detect_sum = np.sum(detect_traces, axis=0)

        if templates is not None and len(templates) > 1:
            # Get the template window on summed traces
            template_info = templates.loc[template_index]
            template_start_time = UTCDateTime(template_info['datetime'])
            start_window_template = template_start_time
            end_window_template = start_window_template + window_size
            windowed_traces_template = st.slice(starttime=start_window_template, endtime=end_window_template)

            # Manually sum the traces within the template time window
            summed_trace_template = np.sum([trace.data for trace in windowed_traces_template], axis=0)

            # Plot the summed traces for the template and the detection
            plot_combined_traces(window_size, detect_sum, summed_trace_template, template_index, network,
                                 channel_prefix, date_of_interest, base_dir, num_detections=len(detect_traces))

        elif len(templates) == 1:
            # Specific template for the comparison
            template_index = templates.index.astype(int).tolist()[0]
            detect_summed_traces = {template_index: detect_summed_traces[template_index]}
            detect_traces = detect_summed_traces[template_index]
            detect_sum = np.sum(detect_traces, axis=0)

            # Get the template window on summed traces
            template_info = templates.loc[template_index]
            template_start_time = UTCDateTime(template_info['datetime'])
            start_window_template = template_start_time
            end_window_template = start_window_template + window_size
            windowed_traces_template = st.slice(starttime=start_window_template, endtime=end_window_template)

            # Manually sum the traces within the template time window
            summed_trace_template = np.sum([trace.data for trace in windowed_traces_template], axis=0)

            # Plot the summed traces for the template and the detection
            plot_combined_traces(window_size, detect_sum, summed_trace_template, template_index, network,
                                 channel_prefix, date_of_interest, base_dir, num_detections=len(detect_traces))
            break

def plot_summed_traces_PB(stas, channels, window_size, network, startdate, date_of_interest, base_dir, templates=None):
    """
    Preprocess PB network seismic data and plot summed traces around detected events.

    Parameters:
        stas (list): List of station names.
        channels (list): List of channel names.
        window_size (int): Time window size in seconds.
        network (str): Network identifier.
        startdate (datetime): Start date of the seismic data collection in 'datetime' format.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
        base_dir (str): The base directory for file paths.
        templates (pd.DataFrame, optional): DataFrame containing template information.
            Defaults to None.

    Returns:
        None
    """
    # Get the data (so it can be called outside of crosscorr.py)
    st = Stream(traces=get_traces_PB(stas, channels, startdate, network, base_dir))
    channel_prefix = channels[0][:2]

    st, stas = process_data(st, stas, locs=None, sampling_rate=80, freqmin=2.0, freqmax=8.0)

    # Define the path to the detections file based on the provided date_of_interest
    output_file_path = os.path.join(base_dir, 'plots', f"{network} {channel_prefix} {date_of_interest}", 'output.txt')
    detections_df = pd.read_csv(output_file_path)

    # Create a dictionary to store summed traces for each template
    detect_summed_traces = {}

    # Iterate through the detections_df
    for _, detection in detections_df.iterrows():

        # Get the template index for the detection
        template_index = detection['templ']

        # Define the time window
        start_time = UTCDateTime(detection['starttime'])
        start_window = start_time
        end_window = start_time + window_size

        # Extract the traces within the time window
        windowed_traces = st.slice(starttime=start_window, endtime=end_window)

        # Manually sum the traces within the time window
        summed_traces = np.sum([trace.data for trace in windowed_traces], axis=0)

        # Append the summed trace to the template's list
        detect_summed_traces.setdefault(template_index, []).append(summed_traces)

    # Sum the traces for each template and plot
    for template_index, detect_traces in detect_summed_traces.items():
        # Sum the traces for the template
        detect_sum = np.sum(detect_traces, axis=0)

        if templates is not None and len(templates) > 1:
            # Get the template window on summed traces
            template_info = templates.loc[template_index]
            template_start_time = UTCDateTime(template_info['datetime'])
            start_window_template = template_start_time
            end_window_template = start_window_template + window_size
            windowed_traces_template = st.slice(starttime=start_window_template, endtime=end_window_template)

            # Manually sum the traces within the template time window
            summed_trace_template = np.sum([trace.data for trace in windowed_traces_template], axis=0)

            # Plot the summed traces for the template and the detection
            plot_combined_traces(window_size, detect_sum, summed_trace_template, template_index, network,
                                 channel_prefix, date_of_interest, base_dir, num_detections=len(detect_traces))

        elif len(templates) == 1:
            # Specific template for the comparison
            template_index = templates.index.astype(int).tolist()[0]
            detect_summed_traces = {template_index: detect_summed_traces[template_index]}
            detect_traces = detect_summed_traces[template_index]
            detect_sum = np.sum(detect_traces, axis=0)

            # Get the template window on summed traces
            template_info = templates.loc[template_index]
            template_start_time = UTCDateTime(template_info['datetime'])
            start_window_template = template_start_time
            end_window_template = start_window_template + window_size
            windowed_traces_template = st.slice(starttime=start_window_template, endtime=end_window_template)

            # Manually sum the traces within the template time window
            summed_trace_template = np.sum([trace.data for trace in windowed_traces_template], axis=0)

            # Plot the summed traces for the template and the detection
            plot_combined_traces(window_size, detect_sum, summed_trace_template, template_index, network,
                                 channel_prefix, date_of_interest, base_dir, num_detections=len(detect_traces))
            break

def plot_combined_traces(window_size, detect_sum, summed_trace_template, template_index, network, channel_prefix, date_of_interest, base_dir, num_detections):
    """
    Plot the combined traces for a detection and its corresponding template.

    Parameters:
        window_size (int): Time window size in seconds.
        detect_sum (numpy.ndarray): Summed trace for the detection.
        summed_trace_template (numpy.ndarray): Summed trace for the template.
        template_index (int): Index of the template.
        network (str): Network identifier.
        channel_prefix (str): Prefix of the channel.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
        base_dir (str): The base directory for file paths.
        num_detections (int): Number of detections in the stack.

   Returns:
        None
    """
    ## 1st figure
    # Plot the summed traces for the template and the detections
    _, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the summed trace for the detection
    num_points_detection = len(detect_sum)
    times_detection = np.linspace(0, window_size, num_points_detection, endpoint=False)
    axs[0].plot(times_detection, detect_sum, label=f'Stack of {num_detections} detections', linestyle='-', color='C0')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # Plot the summed trace for the template
    num_points_template = len(summed_trace_template)
    times_template = np.linspace(0, window_size, num_points_template, endpoint=False)
    axs[1].plot(times_template, summed_trace_template, label=f'Template {template_index}', linestyle='--', color='C1')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)

    # Add legend for the entire figure
    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')

    # Add this line to ensure proper layout
    plt.tight_layout()

    # Save the plot as an image file
    save_path = os.path.join(base_dir, 'plots',
                             f'stack_vs_templ_net{network}_cha{channel_prefix}_templ{template_index}_{date_of_interest}.png')
    plt.savefig(save_path)
    plt.close()

    ## 2nd figure
    # Plot only the summed traces of detections
    plt.figure(figsize=(10, 6))

    num_points_detection = len(detect_sum)
    times_detection = np.linspace(0, window_size, num_points_detection, endpoint=False)
    plt.plot(times_detection, detect_sum, label=f'Stack of {num_detections} detections', linestyle='-', color='C0')

    # Label the x-axis, y-axis, and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Stack of {num_detections} detections net{network}_cha{channel_prefix}_templ{template_index}_{date_of_interest}')
    plt.xlim(0, window_size)
    plt.grid(True)

    # Save the plot as an image file
    save_path = os.path.join(base_dir, 'plots',
                             f'stack_net{network}_cha{channel_prefix}_templ{template_index}_{date_of_interest}.png')
    plt.savefig(save_path)
    plt.show()
