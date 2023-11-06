#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fcts
import datetime
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client

# Define the data parameters
nets = ['CN', 'CN']
stas = ['PFB', 'YOUB']
cha = ['HHN', 'HHE', 'HHZ']
stas = ['LZB','SNB','NLLB','PGC'] 
cha = ['BHN','BHE','BHZ']

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

    # Initialize an empty array to store the cumulative sum
    cumulative_data = None

    for net, sta in zip(nets, stas):
        for component in cha:
            # Construct the filename based on the network, station, and component
            if net in ['PB', 'UW']:
                filename = f'C:/Users/papin/Desktop/phd/data/seed/{yr}/Data/{sta}/{sta}.{net}.{yr}.{day}'
            elif net in ['CN', 'NTKA']:
                filename = f'C:/Users/papin/Desktop/phd/data/seed/{yr}{mth}{tod}.{net}.{sta}..{component}.mseed'

            # Read the file
            stream = read(filename)
            stream.merge(method=fcts.merge_method(skip_on_gaps), fill_value=0)
            trace = stream[0]

            # Define the start and end times
            starttime = UTCDateTime(date) + datetime.timedelta(hours=8) + datetime.timedelta(minutes=0)
            endtime = starttime + datetime.timedelta(minutes=60)

            # Trim the trace
            trace = trace.trim(starttime=starttime, endtime=endtime)

            # Get the data from the trace
            data = trace.data

            # Initialize or add to the cumulative data
            if cumulative_data is None:
                cumulative_data = data
            else:
                cumulative_data += data

    # Create a new trace with the cumulative data
    cumulative_trace = trace.copy()
    cumulative_trace.data = cumulative_data
    # Hide the default Obspy legend
    cumulative_trace.stats['station'] = ""
    cumulative_trace.stats['channel'] = cumulative_trace.stats['channel'][:2]
    # Plot the cumulative trace
    cumulative_trace.plot()
    plt.show()
