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
from obspy import UTCDateTime
from obspy.core import Stream, read
from datetime import datetime

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

def plot_data(date_of_interest, stas, channels, network):
    """
    Plot seismic station data for a specific date.

    Parameters:
        date_of_interest (str): The date in the format 'YYYYMMDD' 
        for which you want to plot the data.
        stas (list of str): List of station names.
        channels (list of str): List of channel names.
        network (str): Network code ('CN' or 'PB').

    This function loads seismic data from the specified stations and channels 
    for the given date, preprocesses the data, and plots the normalized traces
    with an offset for visualization.

    Returns:
        None
    """
    # Initialize an empty Stream to hold the seismic data
    st = Stream()

    if network == 'CN':
        # Load seismic data for the specified stations and channels (CN network)
        for sta in stas:
            for cha in channels:
                path = "C:/Users/papin/Desktop/phd/data/seed"
                file = f"{path}/{date_of_interest}.CN.{sta}..{cha}.mseed"
                try:
                    tr = read(file)[0]
                    st += tr
                except FileNotFoundError:
                    print(f"File {file} not found.")
    elif network == 'PB':
        # Load seismic data for the specified stations and channels (PB network)
        for sta in stas:
            startdate = datetime.strptime(date_of_interest, "%Y%m%d")
            day_of_year = startdate.timetuple().tm_yday
            year = startdate.timetuple().tm_year
            path = "C:/Users/papin/Desktop/phd/data/seed"
            file = f"{path}/{sta}.{network}.{year}.{day_of_year}"
            try:
                for cha in range(len(channels)):
                    tr = read(file)[cha]
                    st += tr
            except FileNotFoundError:
                print(f"File {file} not found.")
    
    # Preprocessing: Interpolation, trimming, detrending, and filtering
    start = max(tr.stats.starttime for tr in st)
    st.interpolate(sampling_rate=80, starttime=start) # Can be modified
    st.detrend(type='simple')
    st.filter("bandpass", freqmin=1.0, freqmax=10.0) # Can be modified
    
    plt.figure(figsize=(15, 5))
    nb = 10 # Distance between plots
    offset = len(stas) * len(channels) * nb
    
    # Get the start date from the first trace in the stream
    start_date = st[0].stats.starttime.strftime("%Y%m%d")
    
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
    plt.savefig(f'C:/Users/papin/Desktop/phd/plots/data_plot_{start_date}.png')
    plt.close()

def get_traces(stas, channels, date_of_interest): # Use less of memory with the yield
    for cha in channels:
        for sta in stas:
            path = "C:/Users/papin/Desktop/phd/data/seed"
            file = f"{path}/{date_of_interest}.CN.{sta}..{cha}.mseed"
            try:
                tr = read(file)[0]
                # print("Loaded data:", tr)
                yield tr
            except FileNotFoundError:
                print(f"File {file} not found.")

def plot_summed_traces(stas, channels, window_size, network, date_of_interest):
    """
    Preprocess seismic data and plot summed traces around detected events.

    Args:
        stas (list): List of station names.
        channels (list): List of channel names.
        window_size (int): Time window size in seconds.
        network (str): Network identifier.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
    """
    # Get the data (so it can be called outside of crosscorr.py)
    st = Stream(traces=get_traces(stas, channels, date_of_interest))
    channel_prefix = channels[0][:2]
    
    # Preprocessing: Interpolation, trimming, detrending, and filtering
    start = max(tr.stats.starttime for tr in st)
    end = min(tr.stats.endtime for tr in st)
    for tr in st:
        tr.trim(starttime=start, endtime=end, fill_value=0)
        tr.interpolate(sampling_rate=80, starttime=start)
        tr.detrend(type='simple')
        tr.filter("bandpass", freqmin=1.0, freqmax=10.0)

    # Define the path to the detections file based on the provided date_of_interest
    output_file_path = f"C:/Users/papin/Desktop/phd/plots/{network} {channel_prefix} {date_of_interest}/output.txt"

    # Read the output file that contains information about detected events
    detections_df = pd.read_csv(output_file_path)

    # Create a list to store the summed traces
    summed_traces = []

    # Iterate through the detections_df
    for index, detection in detections_df.iterrows():
        
        # if index==244:
        # Get the start time from the dataframe and convert it to UTCDateTime
        start_time = UTCDateTime(detection['starttime'])

        # Define the time window using the start time
        start_window = start_time - window_size
        end_window = start_time + window_size

        # Extract the traces within the time window
        windowed_traces = st.slice(starttime=start_window, endtime=end_window)

        # Manually sum the traces within the time window
        summed_trace = windowed_traces[0].copy()  # Initialize with the first trace
        for trace in windowed_traces[1:]:
            summed_trace.data += trace.data

        # Append the summed trace to the list
        summed_traces.append(summed_trace)

    # Plot the summed traces
    for index, summed_trace in enumerate(summed_traces):
        # Create a new figure for each summed trace
        plt.figure(figsize=(10, 6))

        # Get the number of data points in the summed trace
        num_points = len(summed_trace.data)

        # Calculate the time values for the x-axis based on the sample rate
        sample_rate = summed_trace.stats.sampling_rate
        times = [(i - num_points // 2) / sample_rate for i in range(num_points)]

        # Plot the summed trace with time on the x-axis and amplitude on the y-axis
        plt.plot(times, summed_trace.data)

        # Customize the plot
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Summed Traces for Detection {index + 1}')

        # Set x-axis limits and center the x-axis at 0
        plt.xlim(-num_points // 2 / sample_rate, num_points // 2 / sample_rate)

        # Save the plot as an image file
        save_path = (f'C:/Users/papin/Desktop/phd/plots/'
            f'sum_traces_net{network}_cha{channel_prefix}_det{index + 1}.png'
        )
        plt.savefig(save_path)

        # Show the plot for this summed trace
        plt.grid(True)
        plt.close()

def get_traces_PB(stas, channels, startdate, network):
    for sta in stas:
        day_of_year = startdate.timetuple().tm_yday
        year = startdate.timetuple().tm_year
        path = "C:/Users/papin/Desktop/phd/data/seed"
        file = f"{path}/{sta}.{network}.{year}.{day_of_year}"
        try:
            for cha in range(len(channels)):
                tr = read(file)[cha]
                yield tr
        except FileNotFoundError:
            print(f"File {file} not found.")
            
def plot_summed_traces_PB(stas, channels, window_size, network, startdate, date_of_interest):
    """
    Preprocess seismic data and plot summed traces around detected events.

    Args:
        stas (list): List of station names.
        channels (list): List of channel names.
        window_size (int): Time window size in seconds.
        network (str): Network identifier.
        startdate (datetime): Date of interest in datetime format.
        date_of_interest (str): Date of interest in 'YYYYMMDD' format.
    """
    # Get the data (so it can be called outside of crosscorr.py)
    st = Stream(traces=get_traces_PB(stas, channels, startdate, network))
    channel_prefix = channels[0][:2]
    
    # Preprocessing: Interpolation, trimming, detrending, and filtering
    start = max(tr.stats.starttime for tr in st)
    end = min(tr.stats.endtime for tr in st)
    for tr in st:
        tr.trim(starttime=start, endtime=end, fill_value=0)
        tr.interpolate(sampling_rate=80, starttime=start)
        tr.detrend(type='simple')
        tr.filter("bandpass", freqmin=1.0, freqmax=10.0)

    # Define the path to the detections file based on the provided date_of_interest
    output_file_path = f"C:/Users/papin/Desktop/phd/plots/{network} {channel_prefix} {date_of_interest}/output.txt"

    # Read the output file that contains information about detected events
    detections_df = pd.read_csv(output_file_path)

    # Create a list to store the summed traces
    summed_traces = []

    # Iterate through the detections_df
    for index, detection in detections_df.iterrows():
        # Get the start time from the dataframe and convert it to UTCDateTime
        start_time = UTCDateTime(detection['starttime'])

        # Define the time window using the start time
        start_window = start_time - window_size
        end_window = start_time + window_size

        # Extract the traces within the time window
        windowed_traces = st.slice(starttime=start_window, endtime=end_window)

        # Manually sum the traces within the time window
        summed_trace = windowed_traces[0].copy()  # Initialize with the first trace
        for trace in windowed_traces[1:]:
            summed_trace.data += trace.data

        # Append the summed trace to the list
        summed_traces.append(summed_trace)

    # Plot the summed traces
    for index, summed_trace in enumerate(summed_traces):
        # Create a new figure for each summed trace
        plt.figure(figsize=(10, 6))

        # Get the number of data points in the summed trace
        num_points = len(summed_trace.data)

        # Calculate the time values for the x-axis based on the sample rate
        sample_rate = summed_trace.stats.sampling_rate
        times = [(i - num_points // 2) / sample_rate for i in range(num_points)]

        # Plot the summed trace with time on the x-axis and amplitude on the y-axis
        plt.plot(times, summed_trace.data)

        # Customize the plot
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Summed Traces for Detection {index + 1}')

        # Set x-axis limits and center the x-axis at 0
        plt.xlim(-num_points // 2 / sample_rate, num_points // 2 / sample_rate)

        # Save the plot as an image file
        save_path = (f'C:/Users/papin/Desktop/phd/plots/'
            f'sum_traces_net{network}_cha{channel_prefix}_det{index + 1}.png'
        )
        plt.savefig(save_path)

        # Show the plot for this summed trace
        plt.grid(True)
        plt.close()

################################## NOT USED ##################################

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

def plot_crosscorrelation(xcorrmean, thresh, mad, st, stream_duration, 
                          crosscorr_combination, date_of_interest):
    ### DOESN'T WORK
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

def calculate_time_delays(templates_df, output_file_path):
    # Load the templates DataFrame and reset the index
    templates_df.reset_index(inplace=True)
    
    # Read the output file that contains information about detected events
    detections_df = pd.read_csv(output_file_path)

    # Initialize a list to store time delays
    time_delays = []

    # Iterate through the detected events
    for _, detection in detections_df.iterrows():
        # Find the corresponding template for the detected event
        template = templates_df[templates_df['Index'] == detection['templ']].iloc[0]
         
        # Calculate the time delay by subtracting the template's primary event time
        # from the time of the detected event
        detected_event_time = UTCDateTime(detection['starttime'])
        template_event_time = UTCDateTime(template['OT'])
        time_delay = detected_event_time - template_event_time
        
        # Append the time delay to the list
        time_delays.append(time_delay)

    return time_delays

def calculate_time_delays_and_arrival_times(templates_df, output_file_path, Vp, Vs):
    # Load the templates DataFrame and reset the index
    templates_df.reset_index(inplace=True)
    
    # Read the output file that contains information about detected events
    detections_df = pd.read_csv(output_file_path)

    # List to store the calculated time delays and arrival times
    time_delays = []
    arrival_times = []

    for _, detection in detections_df.iterrows():
        # Find the corresponding template for the detected event
        template = templates_df[templates_df['Index'] == detection['templ']].iloc[0]

        # Get the times of the template and the new event 
        template_event_time = UTCDateTime(template['OT'])
        detected_event_time = UTCDateTime(detection['starttime'])

        # Retrieve the depth of the primary event from the template
        D_primary = template['depth']

        # Calculate the time delay by subtracting the template's primary event time
        # from the time of the detected event
        time_delay = detected_event_time - template_event_time

        # Calculate the P-wave and S-wave arrival times 
        p_wave_arrival = template_event_time + time_delay + (D_primary / Vp) ### WRONG
        s_wave_arrival = template_event_time + time_delay + (D_primary / Vs) ### WRONG

        # Append the calculated time delay and arrival times to their respective lists
        time_delays.append(time_delay)
        arrival_times.append((p_wave_arrival, s_wave_arrival))

    return time_delays, arrival_times

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
