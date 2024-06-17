#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:48:37 2023

This module provides functions for seismic data processing, visualization, and analysis.
The functions cover loading and preprocessing data, plotting seismic traces, creating
summed traces around detected events, and more.

Functions:
- build_file_path: Generates file names for plots.
- get_traces: Loads seismic data for specified stations, channels, and date.
- process_data: Preprocess seismic data, including interpolation, trimming,
  detrending, and filtering.
- process_streams: Filters the seismic traces for the first templates.
- remove_stations: Removes the pairs of station and channel with bad data.
- create_window: Creates windows for the template matching based on the arrival
  times.
- create_parameters: Creates the parameters needed for the windows.
- create_arrival_times: Creates a text file containing arrival times based 
  on provided templates and travel time data.
- select_random_templates: Selects a number of random templates for 1 day.
- check_length: Matches the length of variables for traces with different
  numbers of samples (usually just 1 because of the different networks)
- check_xcorr_template: Checks if there aren't nan values in the cross-correlation 
  function and counts how many stations got the coefficients for the normalization.
- remove_crosscorr: If there is a gap in the data, cut it 2 min before to avoid
  any issues with the limits. 
- check_xcorr_full: Checksthe number of channels not available for every point 
  of the function and remove if necessary; updates the crosscorr and mask values.
- check_data: Checks the quality of the data. In this case, used in the function
  plot_stacks to ensure we are stacking correct data.
- plot_data: Plots seismic station data for a specific date, including normalized
  traces with an offset.
- plot_crosscorr: Plots the resulting cross-correlation function for 1 template
  on the network.
- plot_template: Plots the template for each station/channel combination.
- plot_stacks: Plots the combined traces for all detections of a template for all
  station/channel combination.

@author: papin

As of 17/06/24.
"""

import os
import csv
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.core import read, Stream

# ========== Files Management ==========

def build_file_path(base_dir, folder, diff, name, prefix, day):
    """
    Build a file path for saving plots (related to plot_template, plot_stacks,
    plot_crosscorr in crosscorr_tools module).

    Parameters:
        base_dir (str): The base directory where the file will be saved.
        folder (str): The subfolder within the base directory for organization.
        name (str): The base name of the file (usually number of the template).
        prefix (str): A prefix to be included in the file name.
        day (str): Day of the template event.
    """
    return os.path.join(base_dir, f'plots/{folder}/{diff}/{name}_{prefix}_{day}.png')

# ========== Data Loading & Process Streams ==========

def get_traces(network_config, dates_of_interest, base_dir, which=None):
    """
    Retrieve seismic traces for specified stations and channels on a given dates.

    Parameters:
        network_config (dict): A dictionary containing configuration details for each network.
        dates_of_interest (str): The dates in the format 'YYYYMMDD'
        base_dir (str): The base directory for file paths.
        which (str or None): If the code is run on talapas, None value will mean
        the code is running in local.

    Return:
        st (obspy.core.Stream): Seismic data streams.
    """
    # Initialize the stream, stations list, and pairs list
    st = Stream()
    pairs = []
    path = "/projects/amt/shared/" if which else os.path.join(base_dir, 'data', 'seed')

    # Talapas version just goes to search for every streams we ask for
    if which:
        for network, config in network_config.items():
            pathdir = os.path.join(path, "cascadia_" + network)
            stations = config['stations']
            for sta in stations:
                for date in dates_of_interest:
                    pattern = os.path.join(pathdir, f"{date}.*.{sta}.*")
                    stream_to_read = glob.glob(pattern, recursive=True)
                    if not stream_to_read:
                        print(f"File not found for {network} network, station {sta}, date {date}.")
                    for file_path in stream_to_read:
                        st += read(file_path)
    # In local we know the structure of the folders/files
    else:
        for date in dates_of_interest:
            for network, config in network_config.items():
                stations = config['stations']
                channels = config['channels']
                filename_pattern = config['filename_pattern']
                for sta in stations:
                    try:
                        if network == 'PB' or sta == 'VGZ':
                            dt = datetime.strptime(date, '%Y%m%d')
                            julian_day = (dt - datetime(dt.year, 1, 1)).days + 1
                            file = os.path.join(
                                path, filename_pattern.format(
                                    station=sta, year=dt.year, julian_day=julian_day))
                            st += read(file)[:3]
                        else:
                            for cha in channels:
                                file = os.path.join(
                                    path, filename_pattern.format(
                                        date=date, station=sta, channel=cha))
                                st += read(file)
                    except FileNotFoundError:
                        print(f"File not found for {network} network, station {sta}, date {date}.")

    # Merge all data by stations and channels (merge is supposed to already do cleanup)
    st.merge(method=0, fill_value=0)
    st._cleanup()

    # Channels used
    for tr in st:
        pairs.append(tr.id[3:])

    return st, pairs

def process_data(st, startdate, enddate, sampling_rate, freqmin, freqmax):
    """
    Preprocess seismic data.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        startdate (datetime): Beginning of the streams for trimming.
        enddate (datetime): End of the streams for trimming.
        sampling_rate (float): Sampling rate for interpolation.
        freqmin (float): Minimum frequency for bandpass filtering.
        freqmax (float): Maximum frequency for bandpass filtering.

    Return:
        st (obspy.core.Stream): Seismic data streams.
    """
    # Initialize start and end with values that will be updated
    starttime = UTCDateTime(startdate)
    endtime = UTCDateTime(enddate)

    # Preprocessing: Interpolation, trimming, detrending, and filtering
    cpt=0
    for tr in st:
        tr.detrend('linear')
        tr.trim(starttime=starttime, endtime=endtime, pad=1,
                fill_value=0, nearest_sample=False)
        if tr.stats.sampling_rate != 100.0:
            tr.interpolate(sampling_rate=100.0, starttime=starttime, endtime=endtime)
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
        tr.interpolate(sampling_rate=sampling_rate)
        cpt+=1
        print(f'{cpt} out of {len(st)} traces processed (almost there!)')

    return st

def process_streams(st, template_stats, A):
    """
    Process seismic traces and create a new Stream and a list of station-channel pairs.
    It gets which stations detected the event and filter st to keep only those ones.
    
    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        template_stats (dict): Metadata template information.
        A (dict): Dictionary containing additional information based on the template.

    Returns:
        st2 (obspy.core.Stream): Filtered seismic data streams based on the template.
        pairs2 (list): List of pairs corresponding to each trace in st2.
    """
    starttime = str(template_stats['starttime'])

    # Extract information from the template
    templ_info = A[starttime]
    stas = templ_info['sta']

    # Filter streams based on station codes
    st2 = Stream()
    for tr in st:
        # Extract station code from trace id
        station_code = tr.stats.network + '.' + tr.stats.station
        # Check if the station code is in the list of desired stations
        if station_code in stas:
            st2 += tr

    # Create a list of station-channel pairs for st2
    pairs2 = [f"{tr.stats.station}.{tr.stats.channel}" for tr in st2]

    return st2, pairs2

def remove_stations(st, pairs, tr_remove):
    """
    Remove stations from the stream list and their corresponding pairs. The 
    tr_remove might be entirely based on a abritrary decision while looking
    at the streams plot or else.

    Parameters:
        st (list): List of stream objects.
        pairs (list): List of corresponding pairs to the streams.
        tr_remove (list, optional): List of station names to be removed. Default is None.

    Returns:
        st (list): Modified list of stream objects.
        pairs (list): Modified list of corresponding pairs to the streams.

    Note:
        Usually tr_remove being None will refer to the first run of the script
        when we don't really know which stations can't be used or the process.
    """
    if tr_remove:
        length=len(st)
        idx_remove = [i for i, tr in enumerate(st)
                      if tr.stats.station + '..' + tr.stats.channel in tr_remove]
        if idx_remove:
            for idx in sorted(idx_remove, reverse=True):
                st.pop(idx)
                del pairs[idx]
        print(f"{length-len(st)} traces have been removed from the original configurations.")
    else:
        print("No stations have been removed from the original configurations.")

    return st, pairs

# ========== Template Matching Parameters ==========

def create_window(templates, stas, base_dir, diff):
    """
    Creates windows for cross-correlation based on arrival times.

    Parameters:
        templates (DataFrame): All templates for the cross-correlation.
        stas (list): List of the stations used.
        base_dir (str): The base directory for file paths.

    Return:
        windows (list): List of DataFrames containing winsdow data.

    """
    templates_info = create_parameters(templates, stas, base_dir, diff)
    # Initialize lists to store the windows
    templ_idx_ = []
    begin_interval_S_ = []
    begin_interval_P_ = []
    nb_stas_P_ = []
    nb_stas_S_ = []
    for idx, _ in templates.iterrows():
        templ_idx_.append(idx)
        param = templates_info.get(str(idx), {})
        begin_interval_P = param.get('interval_lower_P', 0)
        begin_interval_S = param.get('interval_upper_S', 0)
        begin_interval_P_.append(begin_interval_P)
        begin_interval_S_.append(begin_interval_S)
        nb_stas_P = param.get('count_stations_P', 0)
        nb_stas_P_.append(nb_stas_P)
        nb_stas_S = param.get('count_stations_S', 0)
        nb_stas_S_.append(nb_stas_S)
    windows = pd.DataFrame({'template': templ_idx_,
                            'timedelta_P': begin_interval_P_,
                            'timedelta_S': begin_interval_S_,
                            'nb_stas_P': nb_stas_P_,
                            'nb_stas_S': nb_stas_S_})
    windows.set_index('template', inplace=True)
    return windows

def create_parameters(templates, stas, base_dir, diff):
    """
    Create parameters for seismic templates based on arrival times.
    
    Parameters:
        templates (DataFrame): All templates for the cross-correlation.
        stas (list): List of the stations used.
        base_dir (str): The base directory for file paths.
    
    Return:
        templates_info (dict): A dictionary containing information about each seismic template.
            Keys are template IDs and values are dictionaries with the following keys:
            - 'min_p_wave': The minimum P-wave arrival time among selected stations.
            - 'interval choosen': Description of the chosen time interval for analysis.
            - 'how many stas in interval': Number of stations within the chosen interval.
            - 'stations': List of station codes within the chosen interval.
    
    NB: Every computation is settled for windows of 10 sec so it has to be 
    changed here for correct windows.
    """
    # Read the arrival times
    filename = create_arrival_times(templates, base_dir, diff)
    AT=os.path.join(os.path.join(base_dir, filename))

    # Read the data from the file
    with open(AT, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)
        data = [line for line in reader]

    # Choose the same data for the choosen templates
    templates_list = [str(num) for num in templates.index]
    templates_info = {}

    # Iterate over each family
    for template in templates_list:
        # Choosing stations
        template_data = [line for line in data if line[0] == template and line[1] in stas]

        # Extract station names and corresponding P and S wave times
        stations = [line[1] for line in template_data]
        p_wave_times = [float(line[2]) for line in template_data]
        s_wave_times = [float(line[3]) for line in template_data]

        # Sort stations by P-wave times in reverse order while corresponding S-wave times
        sorted_data = sorted(zip(p_wave_times, stations, s_wave_times), reverse=True)
        p_wave_times, stations, s_wave_times = zip(*sorted_data)

        # Maths (more in create_arrival_times)
        min_p_wave = round(min(p_wave_times), 3)
        percentile_75_S_wave = round(np.percentile(s_wave_times, 75), 3)

        # How many stations fit in the interval
        interval_lower_P = np.floor(min_p_wave * 2) / 2
        interval_upper_P = interval_lower_P + 10
        count_stations_P = sum(1 for i in range(len(stations))
                               if interval_lower_P <= p_wave_times[i] <= interval_upper_P
                               and interval_lower_P <= s_wave_times[i] <= interval_upper_P)
        interval_upper_S = np.ceil(percentile_75_S_wave * 2) / 2
        interval_lower_S = interval_upper_S - 10
        count_stations_S = sum(1 for i in range(len(stations))
                               if interval_lower_S <= p_wave_times[i] <= interval_upper_S
                               and interval_lower_S <= s_wave_times[i] <= interval_upper_S)

        # Define the parameters
        templates_info[template] = {
            'interval_lower_P': interval_lower_P,
            'interval_upper_S': interval_upper_S,
            'count_stations_P': count_stations_P,
            'count_stations_S': count_stations_S,
            'stations': stations
        }
        
        ## Plotting part
        fig, ax = plt.subplots(figsize=(8, 6))
        for station in stations:
            ax.plot([0, 30], [station, station], color='grey', linestyle='-', linewidth=1)
        ax.scatter(p_wave_times, stations, color='blue', marker='o', label='P Wave')
        ax.scatter(s_wave_times, stations, color='red', marker='o', label='S Wave')
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Station')
        ax.set_title(f'Wave Times for template {template} - count={count_stations_P}')
        for s_wave, p_wave, station in zip(s_wave_times, p_wave_times, stations):
            ax.plot([p_wave, s_wave], [station, station], color='black', linestyle='-', linewidth=2)
        # Add vertical lines on the x-axis
        ax.axvline(x=interval_lower_P, color='blue', linestyle='--', label='Interval P')
        ax.legend(loc='upper right')
        ax.axvline(x=interval_upper_P, color='blue', linestyle='--')
        windows_plot_filename = build_file_path(
            base_dir, 'SSE_2005', f'{diff}', f'templ{template}', 'AT', f'{diff}')
        plt.tight_layout()
        plt.savefig(windows_plot_filename, dpi=300)
        plt.close(fig)

    return templates_info

def create_arrival_times(templates, base_dir, diff):
    """
    Create a text file containing arrival times based on provided templates 
    and travel time data.

    Parameters:
        templates (DataFrame): All templates for the cross-correlation.
        base_dir (str): The base directory for file paths.

    Return:
        output_file_path (str): The file path of the created text file 
        containing arrival times.
    """
    def idx_loc(loc):
        """
        Given a rough location, return the index of the closest loc in grid-nodes.

        Parameters:
            loc (array-like): The coordinates [longitude, latitude, depth].

        Return:
            idx (int): The index of the closest location in the grid-nodes.
        """
        d2 = np.sum((coords[:, :3] - loc)**2, axis=1)
        idx = np.argmin(d2)
        return idx

    # Load travel times from Tim's grid
    TT = np.load(os.path.join(base_dir, "Travel.npy"), allow_pickle=True).item()
    coords = np.array(list(TT['T'].keys()))  # Coordinates of each point on the grid
    sta_phase = TT['sta_phase']  # Station and P/S waves list
    output_file_path = os.path.join(base_dir, f'AT_{diff}.txt')
    templates['index']=templates.index
    # Create the txt file with the arrival times
    coords2 = np.zeros((len(templates), 3))  # Coordinates of each detection
    cpt=0
    for cpt, (i, row) in enumerate(templates.iterrows()):
        coords2[cpt] = [row['lon'], row['lat'], row['depth']]
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Write header
        output_file.write("template,station,P-wave,S-wave,difference,lon_event,lat_event,z_event\n")
        for idx, (_, row) in enumerate(templates.iterrows()):
            lon_event, lat_event, z_event = coords2[idx]
            exact_loc = coords[idx_loc([lon_event, lat_event, z_event])]
            arriv_times = TT['T'][tuple(exact_loc)]
            # Write data for each station and its P- and S-wave arrival times
            for i in range(0, int(len(arriv_times) / 2)):
                sta = sta_phase[i][:-2]
                time_P = arriv_times[i]
                time_S = arriv_times[i + int(len(arriv_times) / 2)]
                diff = abs(time_P - time_S)
                output_file.write(
                    f"{templates['index'].iloc[idx]},{sta},{time_P},{time_S},{diff},"
                    f"{lon_event},{lat_event},{z_event}\n"
                )

    return output_file_path

def select_random_templates(df):
    """
    Selects 20 random templates from a DataFrame.

    Parameters:
        df (DataFrame): The DataFrame containing the templates.

    Return:
        df (DataFrame): A DataFrame containing a subset of randomly selected templates.
    """
    if len(df) <= 20:
        return df

    return df.sample(n=20, random_state=42)

# ========== Verifications for Data ==========

def check_length(xcorr_full, xcorr_template, mask):
    """
    Ensure that the length of the cross-correlation arrays matches.

    Parameters:
        xcorr_full (numpy.ndarray): Array containing the full cross-correlation values.
        xcorr_template (numpy.ndarray): Array containing the cross-correlation 
        values for a template.
        mask (numpy.ndarray): Array containing the mask values.

    Return:
        tuple: A tuple containing the corrected cross-correlation arrays.
    """
    if len(xcorr_template) < len(xcorr_full):
        xcorr_full = xcorr_full[:len(xcorr_template)]
        mask = mask[:len(xcorr_template)]
    elif len(xcorr_template) > len(xcorr_full):
        xcorr_template = xcorr_template[:len(xcorr_full)]
        mask = mask[:len(xcorr_full)]
    return xcorr_full, xcorr_template, mask

def check_xcorr_template(xcorr_template, mask):
    """
    Check the cross-correlation values of a template and update a mask accordingly.

    Parameters:
        xcorr_template (numpy.ndarray): The cross-correlation values of the template.
        mask (numpy.ndarray): An array used to track problematic values.

    Returns:
        xcorr_template: The updated cross-correlation values.
        mask: The updated mask array.

    If the `xcorr_template` contains NaN values, they are replaced with zeros using `np.nan_to_num`.
    Then, the function identifies non-zero elements in `xcorr_template` and updates 
    the corresponding positions in the `mask` array by incrementing their values by 1.

    Examples:
    >>> template = np.array([0.5, 0.3, np.nan, 0.2])
    >>> mask = np.zeros(len(template))
    >>> check_xcorr(template, mask)
    (array([0.5, 0.3, 0. , 0.2]), array([1., 1., 0., 1.]))
    """
    if np.isnan(xcorr_template).any():
        # Put 0 values instead of nan
        xcorr_template = np.nan_to_num(xcorr_template)
        # Get the indexes where the values aren't null
        idx = np.where(xcorr_template!=0) #>=0.00001
        mask[idx]+=1
    else:
        mask+=1
    return xcorr_template, mask

def remove_crosscorr(idx, samples=2400):
    """
    Remove crosscorr values by adding indexes number.
    
    Parameters:
        idx (tuple): The indexes of bad values of crosscorr.
        samples (int): Number of samples to add to idx.

    Returns:
        idx (tuple): The actualized indexes.
    """
    idxx = idx[0]
    idx_set = set(idxx)
    # Find the beginning of each suite
    starts = [idxx[0]]  # The first element is always a start
    for i in range(1, len(idxx)):
        if idxx[i] != idxx[i - 1] + 1:
            starts.append(idxx[i])
    # Add indices before each start to account for 1 minute of samples
    for start in starts:
        if start - samples >= 0:
            idx_set.update(range(start - samples, start))
        else:
            idx_set.update(range(0, start))
    idx=(np.array(sorted(idx_set)),)
    return idx

def check_xcorr_full(xcorr_full, mask):
    """
    Check the cross-correlation values of a template and update a mask accordingly.

    Parameters:
        xcorr_full (numpy.ndarray): The network cross-correlation values of the template.
        mask (numpy.ndarray): An array used to track problematic values.

    Returns:
        xcorr_full: The updated cross-correlation values.
        mask: The updated mask array.

    If the `xcorr_full` has values coming from a small number of stations, we 
    chose to put the values of the function and the mask at 0. This is made to 
    avoid bias on the normalization of each cc fct values, which is depending 
    on the number of stations from which the data is coming. 
    
    NB: For now, the decisions are made based on the networks CN and PO for the 
    2005 SSE so it can be adapted. It is around a 20% channels missing.
    """
    nb_chas = np.max(mask) # Can be different from the number of stations*3
    if nb_chas/3<=6: #PO or CN or C8
        idx = np.where(mask<nb_chas-3)
    elif nb_chas/3<=12: #PO and CN
        idx = np.where(mask<nb_chas-6)
    elif nb_chas/3>12: #PO and CN and C8
        idx = np.where(mask<nb_chas-9)
    if idx[0].size != 0:
        # Remove 2 minutes of data for each values = 0
        idx=remove_crosscorr(idx, samples=2400*2)
        # Remove the crosscorr and mask values
        mask[idx]=0
        xcorr_full[idx]=0
        print('Some values of the cc fct have been removed due to the small '
              'number of available channels for these points.')
    return xcorr_full, mask

def check_data(data):
    """
    Check the quality of seismic data.
    
    Return:
        bool: True if it passes all the checks, otherwise False.
    
    NB: Used in the stacking of the new detections for now.
    """
    if len(data) == 0:
        return False
    if np.any(np.isnan(data)):
        return False
    if np.max(np.abs(data)) == 0:
        return False
    return True

# ========== All Types of Plots ==========

def plot_data(st, pairs, data_plot_filename):
    """
    Plot seismic station data for a specific date. A figure for each
    station. Note that the way the function was made is for a station with its 
    3 channels; if one is removed you'll have an error.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        stas (list of lists): List of lists of station names.
        channels (list): List of channel names.
        data_plot_filename (str): Filename for saving the plot.

    """
    cpt=0
    for i in range(0, len(st), 3):
        plt.figure(figsize=(12, 6))
        nb = 1  # Distance between plots
        offset = len(st[i:i+3]) * nb
        for tr in st[i:i+3]:
            pair=pairs[cpt]
            norm = np.max(np.abs(tr.data))
            plt.plot(tr.times(), tr.data / norm + offset, label=f'{pair}')
            offset -= nb
            cpt+=1
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Normalized Data + Offset', fontsize=14)
        plt.title(f'Data from {tr.stats.starttime.date} to {tr.stats.endtime.date}', fontsize=16)
        plt.yticks(np.arange(len(st[i:i+3])) * nb+nb, pairs[cpt-3:cpt][::-1], fontsize=12)
        plt.grid(True)
        plt.xlim(0, max(tr.times()))
        plt.ylim(0, len(st[i:i+3]) * nb + nb)
        plt.tight_layout()
        plt.savefig(f"{data_plot_filename}_{tr.stats.station}.png", dpi=300)
        plt.close()

def plot_crosscorr(xcorrmean, thresh, dt, newdect,
                   templ_idx, crosscorr_plot_filename, cpt, mask=False):
    """
    Plots cross-correlation data and saves the plot to a file.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        xcorrmean (numpy.ndarray): The cross-correlation mean values.
        thresh (float): Threshold for the new detections.
        newdect (numpy.ndarray): Indices of the detected events.
        templ_idx (int): Index of the template.
        crosscorr_plot_filename (str): Path to save the plot.
        cpt (int): Number of the iteration for the title.

    """
    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dt * np.arange(len(xcorrmean)), xcorrmean, label='Cross-correlation')
    ax.axhline(thresh, color='red', label='Threshold', alpha=0.7)
    ax.plot(newdect * dt, xcorrmean[newdect], 'kx', label='Detected events')
    if np.any(mask):
        ax_mask = ax.twinx()
        ax_mask.plot(dt * np.arange(len(mask)), mask, color='green', label='Mask', alpha=0.7)
        ax_mask.set_ylabel('Number of channels', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Correlation Coefficient', fontsize=14)
    ax.set_xlim(0, len(xcorrmean)*dt)
    ax.set_title(f'Cross-correlation Function for Template {templ_idx} '
                 f'- Iteration {cpt} - {len(newdect)} detections', fontsize=16)
    # lines, labels = ax.get_legend_handles_labels()
    # if np.any(mask):
    #     lines_mask, labels_mask = ax_mask.get_legend_handles_labels()
    #     ax.legend(lines + lines_mask, labels + labels_mask, loc='center right')
    # else:
    #     ax.legend(lines, labels, loc='center right')
    plt.tight_layout()
    plt.savefig(crosscorr_plot_filename, dpi=300)
    plt.close()

def plot_template(all_template, pairs, time_event, N, sampling_rate,
                  templ_idx, template_plot_filename):
    """
    Plot normalized templates with an offset on the y-axis.

    Parameters:
        all_template (list): List of template traces.
        pairs (list): List of pairs corresponding to each template in all_template.
        time_event (UTCDateTime): The event time for the template.
        N (int): The number of stations that detected the event.
        sampling_rate (float): Sampling rate of the template data.
        templ_idx (int): Index of the template.
        template_plot_filename (str): Filename for saving the plot.

    """
    # Plot each template with an offset on the y-axis
    plt.figure(figsize=(12,6))
    nb = 1  # Distance between plots
    offset = len(all_template) * nb
    x = np.linspace(0, len(all_template[0]) / sampling_rate,
                    len(all_template[0]), endpoint=False)
    for template, pair in zip(all_template, pairs):
        if not check_data(template.data):
            continue
        norm = np.max(np.abs(template.data))
        plt.plot(x, template.data / norm + offset, label=f'{pair}')
        offset -= nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'All Templates for Template {templ_idx} - {time_event} - N={N}', fontsize=16)
    plt.yticks(np.arange(len(pairs)) * nb+nb, pairs[::-1], fontsize=11)
    plt.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    plt.xlim(0, max(x))
    plt.ylim(0, len(pairs) * nb + nb)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(template_plot_filename, dpi=300)
    plt.close()

def plot_stacks(st, newdect, pairs, templ_idx, stack_plot_filename, cpt):
    """
    Plot the combined traces of the detections of the template. Then will be
    used as new templates.

    Parameters:
        st (obspy.core.Stream): Seismic data streams.
        newdect (numpy.ndarray): Indices of the detected events.
        pairs (list): List of pairs corresponding to each trace in st.
        templ_idx (int): Index of the template.
        stack_plot_filename (str): Filename for saving the plot.
        cpt (int): Number of the iteration for the title.

    """

    # Stacking process
    stacked_traces = np.zeros((len(st), int(30 * st[0].stats.sampling_rate)))
    for idx, tr in enumerate(st):
        cptwave=0 # Number of waveforms we don't stack bc of value issues
        all_waveform=[]
        for dect in newdect:
            # Normalize each waveform by its maximum absolute amplitude
            start_time = dect - int(5 * tr.stats.sampling_rate)
            end_time = dect + int(25 * tr.stats.sampling_rate)
            if end_time > len(tr.data):
                continue
            waveform_window = tr.data[start_time:end_time]
            if not check_data(waveform_window):
                cptwave += 1
                continue
            max_abs_value = np.max(np.abs(waveform_window))
            normalized_waveform = waveform_window / max_abs_value
            stacked_traces[idx, :] += normalized_waveform
            all_waveform.append(normalized_waveform)
    stacked_traces /= len(newdect-cptwave)

    # Plot each stack with an offset on the y-axis
    plt.figure(figsize=(12,6))
    nb = 1  # Distance between plots
    offset = len(stacked_traces) * nb
    x = np.linspace(0, 30, len(stacked_traces[0, :]), endpoint=False)
    for i in range(len(stacked_traces)):
        norm = np.max(np.abs(stacked_traces[i,:])) # Same weight for each stack on the figure
        plt.plot(x, stacked_traces[i,:]/norm+offset, label=f'{pairs[i]}')
        offset -= nb
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Normalized Data + Offset', fontsize=14)
    plt.title(f'Stacked Traces for Template {templ_idx} '
              f'- Iteration {cpt} - {len(newdect)} detections', fontsize=16)
    plt.yticks(np.arange(len(pairs))*nb+nb, pairs[::-1], fontsize=11)
    plt.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    plt.xticks([0,5,10,20,30],[-5,0,5,15,25])
    plt.xlim(0,30)
    plt.ylim(0, len(pairs)*nb+nb)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(stack_plot_filename, dpi=300)
    plt.close()
    return stacked_traces

# ========== Old functions that can be reused ==========

# =============================================================================
# def plot_loc(locs, base_dir, events=None):
#     """
#     Create a plot of station locations and template events on a map.
#
#     Parameters:
#         locs (list): A list of tuples, where each tuple contains station name,
#         longitude, and latitude.
#         base_dir (str): The base directory for file paths.
#         events (DataFrame or None): A DataFrame containing event data with
#         columns 'lon','lat','depth', and 'datetime', or None if no events are provided.
#
#     This function creates a scatter plot of station locations and labels them
#     with station names. If events are provided, the function also plots the
#     events and labels them with 'Events'. The resulting plot is saved as
#     'station_events_locations_{date}.png'.
#
#     Returns:
#         None
#     """
#     plt.figure()
#
#     # Separate stations by network and plot them in different colors
#     for name, lon, lat, network in locs:
#         if network == 'CN':
#             plt.plot(lon, lat, 'bo')
#         elif network == 'PB':
#             plt.plot(lon, lat, 'ro')
#         plt.text(lon, lat, name)
#     title = 'Station Locations'
#     filename = 'station_locations.png'
#
#     # Plot events if provided
#     if events:
#         plt.scatter(events['lon'], events['lat'], c='grey', marker='x', label='Events')
#         # plt.colorbar(scatter, label='Depth') #c='events['depth']'
#         date=(UTCDateTime(events['datetime'][0]).date).strftime('%Y%m%d')
#         title = f'Station and Events Locations on {date}'
#         filename = 'station_events_locations_{date}.png'
#
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title(title)
#     plt.savefig(os.path.join(base_dir, 'plots', filename))
#     plt.close()
# =============================================================================

# =============================================================================
# def check_peak_frequency(all_template, sampling_rate=40, frequency_range=(1, 8)):
#     """
#     Function to check if the highest peak in the amplitude spectrum of any trace
#     falls within the frequency range of 1 to 2 Hz, and also plot the amplitude spectrum.
#
#     Parameters:
#         all_template (list of arrays): List containing seismic signal traces.
#         sampling_rate (float): Sampling rate in Hz.
#         frequency_range (tuple): Frequency range (start, stop) in Hz.
#
#     Return:
#         condition_met (bool): False if the condition is met for any trace,
#         True otherwise.
#
#     NB: Possibly can be used in the process so keeping it for now.
#     """
#     start_freq, stop_freq = frequency_range
#     num_traces = len(all_template)
#     condition_met = True
#     for i in range(num_traces):
#         # Execute the FFT on the data
#         tr = all_template[i]
#         max_abs = np.max(np.abs(tr))
#         tr_norm = tr / max_abs
#         n = len(tr_norm)
#         dt = 1 / sampling_rate
#         freq = np.fft.fftfreq(n, dt)
#         amp_spectrum = np.abs(np.fft.fft(tr_norm))
#         freq_mask = (freq >= start_freq) & (freq <= stop_freq)
#         # Sort the amplitudes in descending order and get the corresponding frequencies
#         sorted_indices = np.argsort(amp_spectrum[freq_mask])[::-1]
#         highest_amp_idx = sorted_indices[0]
#         # Check if the highest peak is between 1-2 Hz (most likely noise we don't want)
#         if 1.0 <= freq[freq_mask][highest_amp_idx] < 2.0:
#             condition_met = False
#
#     return condition_met
# =============================================================================

## Old version of create_window to get each station a window depending on S-wave times
# =============================================================================
#     # All S-wave times; the file has to be changed depending on the period
#     AT = pd.read_csv('./arrival_times_tim_2005_SSE.txt', index_col=0)
#     # All times for the windows
#     windows = []
#     for idx, _ in templates.iterrows():
#         templ_idx = idx
#         stas = AT[AT.index == templ_idx]['station'] # Only stations from Travel.npy
#         S_times = AT[AT.index == templ_idx]['S-wave']
#         timedelta = (np.ceil(S_times * 2) / 2) - win_size + 0.5
#         differences = round(AT[AT.index == templ_idx]['difference'],1)
#         PandS = differences.apply(lambda x: 'OK' if x < win_size else 'NO')
#         window_data = pd.concat([stas, S_times, timedelta, PandS], axis=1)
#         window_data.columns.values[-2] = 'timedelta'
#         window_data.columns.values[-1] = 'PandS'
#         windows.append(window_data)
# =============================================================================
