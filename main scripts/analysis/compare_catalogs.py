#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:39:03 2024

This script processes seismic event data from two catalogs: an initial catalog 
containing seismic events' metadata and a secondary catalog of detections. 

The initial catalog (`lfe_svi.txt`) contains seismic event data that helped to 
create the templates for cross-correlation that initied the detections (`output.txt`).

The idea is to verify that the "new detections" actually confirms the data we
have in the initial catalog.

@author: lpapin
"""
from datetime import datetime
import pandas as pd

# Dataframe
startdate=datetime.strptime('20050903', "%Y%m%d")
enddate=datetime.strptime('20050908', "%Y%m%d")

### Bostock catalog; initial templates ###

cata_init = pd.read_csv('lfe_svi.txt', index_col=0, dtype={'date': str, 'hour': str, 'lfe_family':str})
cata_init['date'] = '20' + cata_init['date']
cata_init['date'] = pd.to_datetime(cata_init['date'], format='%Y%m%d')
cata_init['hour'] = cata_init['hour'].str.zfill(2)
cata_init['OT'] = cata_init['date'] + pd.to_timedelta(cata_init['hour'].astype(int), unit='h') + pd.to_timedelta(cata_init['second'], unit='s')
cata_init = cata_init[(cata_init['OT'] >= startdate) & (cata_init['OT'] < enddate)]
cata_init = cata_init.drop(columns=['Mw','hour','second','date'])
cata_init = cata_init.sort_values(by='OT', ascending=True)
cata_init.reset_index(inplace=True)
cata_init.index.name = 'Index'

### Output catalog of detections ###

cata_out=pd.read_csv('/Users/lpapin/Documents/phd/plots/bostock/run 6/output.txt')

## Test 1 template
# cata_out=cata_out[cata_out['templ'] == 0]
# removed_duplicates = cata_out[cata_out.duplicated(subset=['starttime'], keep='last')]
# cata_out_unique = cata_out.drop_duplicates(subset=['starttime'], keep='last')

## Generalize to all cata
# Get unique values of 'templ'
unique_templ_values = cata_out['templ'].unique()
cata_out_unique_all = pd.DataFrame()
removed_duplicates_all = pd.DataFrame()
# Iterate over each unique value of 'templ'
for templ_value in unique_templ_values:
    cata_out_filtered = cata_out[cata_out['templ'] == templ_value]
    removed_duplicates = cata_out_filtered[cata_out_filtered.duplicated(subset=['starttime'], keep='first')]
    removed_duplicates_all = pd.concat([removed_duplicates_all, removed_duplicates])
    cata_out_unique = cata_out_filtered.drop_duplicates(subset=['starttime'], keep='first')
    cata_out_unique_all = pd.concat([cata_out_unique_all, cata_out_unique])

cata_out=cata_out_unique_all
cata_out['OT'] = pd.to_datetime(cata_out['starttime']) - pd.Timedelta(seconds=8)
cata_out.drop(columns=['starttime'], inplace=True)

# Comparision between both catalogs
merged_data = pd.merge(cata_out, cata_init, how='inner', on='OT')
print(merged_data)
new_detect=merged_data[merged_data['crosscorr value'] != 1]
print(new_detect)
