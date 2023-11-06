#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fcts
import datetime
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client

# Define the data parameters
net = 'CN'
sta = 'PFB'
cha = 'HHE'

segm = 3600
skip_on_gaps = False

# Set the start date and number of days to process
start_date = datetime.date(2010, 5, 16)
num_days = 1

# Create an Obspy client
client = Client("IRIS")

for day_offset in range(num_days):
    # Calculate the current date
    date = start_date + datetime.timedelta(days=day_offset)
    
    # Set the day, month, and year
    day = date.timetuple().tm_yday
    yr = str(date.timetuple().tm_year)
    mth = date.strftime('%m')
    tod = date.strftime('%d')

    # Construct the filename based on the network and station
    if net in ['PB', 'UW']:
        filename = f'C:/Users/papin/Desktop/phd/data/seed/{yr}/Data/{sta}/{sta}.{net}.{yr}.{day}'
    elif net in ['CN', 'NTKA']:
        filename = f'C:/Users/papin/Desktop/phd/data/seed/{yr}{mth}{tod}.{net}.{sta}..{cha}.mseed'

    # Read the file
    stream = read(filename)
    stream.merge(method=fcts.merge_method(skip_on_gaps), fill_value=0)
    trace = stream[0]
    sampling_rate = trace.stats.sampling_rate

    # Define the start and end times
    starttime = UTCDateTime(date) + datetime.timedelta(hours=6.15)
    endtime = starttime + datetime.timedelta(minutes=6)

    # Trim the trace
    trace = trace.trim(starttime=starttime, endtime=endtime)

    # Plot the trace
    trace.plot()
    plt.show()
