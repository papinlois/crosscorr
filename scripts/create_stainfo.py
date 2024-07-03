#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:27:11 2024

@author: lpapin
"""

data = """\
STATION  LON        LAT        ELEVATION
SILB     -123.2815  48.6020    76.0
SSIB     -123.3875  48.7558    12.0
TSJB     -123.9885  48.6013    378.0
TWKB     -123.7332  48.6448    128.0
KLNB     -123.5706  48.6611    0.0
"""

# Save to a .dat file
with open('stations_all_4.dat', 'w') as file:
    file.write(data)


import numpy as np
import pandas as pd

sav_sta = {}
sav_lon = []
sav_lat = []
stainfo = pd.read_csv('stations_all_4.dat', sep='\s+')

for i in range(len(stainfo)):
    staname = stainfo.iloc[i]['STATION']  # Just use the station code
    if staname not in sav_sta:
        sav_sta[staname] = [stainfo.iloc[i]['LON'], stainfo.iloc[i]['LAT'], stainfo.iloc[i]['ELEVATION']]
        sav_lon.append(stainfo.iloc[i]['LON'])
        sav_lat.append(stainfo.iloc[i]['LAT'])

np.save('stainfo.npy', sav_sta)
