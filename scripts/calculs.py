# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:51:35 2023

@author: papin
"""

import os
from datetime import datetime

def group_lines_by_starttime_minute(input_file_path, output_file_name):
    """
    Group lines with the same starttime up to the minute 
    and write the results to a file.
    """
    starttime_lines = {}

    with open(input_file_path, "r") as input_file:
        header = input_file.readline()  # Read and store the header line
        for line in input_file:
            parts = line.strip().split(',')
            starttime = parts[0][:17]  # Extract the first 17 characters of the starttime
            if starttime in starttime_lines:
                starttime_lines[starttime].append(line)
            else:
                starttime_lines[starttime] = [line]

    with open(output_file_name, "w") as detect_file:
        for starttime, lines in sorted(starttime_lines.items()):
            if len(lines) > 1:
                detect_file.write(f"Lines with the same starttime ({starttime}):\n")
                for line in sorted(lines):  # Sort the lines by starttime
                    detect_file.write(line)

def group_lines_by_starttime_window(input_file_path, output_file_name, time_range_minutes):
    """
    Group lines with the same starttime within a specified time window 
    and write the results to a file.
    """
    starttime_lines = {}

    with open(input_file_path, "r") as input_file:
        header = input_file.readline()  # Read and store the header line
        for line in input_file:
            parts = line.strip().split(',')
            starttime_str = parts[0][:19]  # Extract the first 19 characters of the starttime (including milliseconds)

            # Handle optional seconds part
            if len(parts[0]) > 19:
                starttime_str += parts[0][19:]  # Append the milliseconds part if present

            try:
                starttime = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S.%f")  # Parse starttime as a datetime object
            except ValueError:
                # Handle case where milliseconds are missing
                starttime = datetime.strptime(starttime_str, "%Y-%m-%dT%H:%M:%S")

            if not starttime_lines:
                starttime_lines[starttime] = [line]
            else:
                for prev_starttime in list(starttime_lines.keys()):
                    time_diff = (starttime - prev_starttime).total_seconds()  # Calculate time difference in seconds
                    if -time_range_minutes * 60 <= time_diff <= time_range_minutes * 60:
                        starttime_lines[prev_starttime].append(line)
                        break
                else:
                    starttime_lines[starttime] = [line]

    with open(output_file_name, "w") as detect_file:
        for starttime, lines in sorted(starttime_lines.items()):
            if len(lines) > 1:
                detect_file.write(f"Lines within -{time_range_minutes} to +{time_range_minutes} minutes of starttime ({starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')}):\n")
                for line in sorted(lines):  # Sort the lines by starttime
                    detect_file.write(line)

# Utilization
group_lines_by_starttime_minute("output.txt", "detect_minute.txt")
group_lines_by_starttime_window("output.txt", "detect_window.txt", time_range_minutes=10)
