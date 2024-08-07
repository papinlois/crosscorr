#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:45:08 2024

@author: lpapin

This script analyzes seismic event data for any period of less than 1 year.

1. A scatter plot on a map displaying the distribution of events over time, 
   with the colormap indicating the day of the year.
2. A 3D scatter plot showing the distribution of events with depth, 
   colored by the day of the year.
3. Monthly seismicity maps illustrating the distribution of events for each 
   month of the year.
4. A scatter plot showing the latitude distribution of events over time, 
   colored by the day of the year.
5. A bar plot depicting the daily event rate throughout the year.

As of 16/05/2024.
"""

from datetime import datetime
import cartopy.crs as ccrs
import crosscorr_tools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator  # Import MaxNLocator for tick locator

## Events (same structure as crosscorr)
startall = datetime.strptime("20050101", "%Y%m%d")
endall = datetime.strptime("20051231", "%Y%m%d")
startdate = datetime.strptime("20050903", "%Y%m%d")
enddate = datetime.strptime("20050925", "%Y%m%d")
templates=pd.read_csv('./EQloc_001_0.1_3_S.txt_withdates', index_col=0)
templates=templates[(templates['residual']<0.5)]
templates['OT'] = pd.to_datetime(templates['OT']) # Formatting 'OT' column as datetime
templates = templates[(templates['OT'] >= startall)
                    & (templates['OT'] < endall)
                    & (templates['residual'] < 0.1)]
templates = templates.drop(columns=['dates', 'residual', 'dt'])
templates.reset_index(inplace=True, drop=True)
templates.index.name = 'Index'
# templates = templates[(templates['OT'] >= startdate)
#                     & (templates['OT'] < enddate)]
# To choose which templates
# templates = templates.sort_values(by='N', ascending=False)
# templates=templates.iloc[1479:1480+1]
# print(templates)
# # 4 events per day with highest N values 
# random_templates = templates.groupby(templates['OT'].dt.date).apply(crosscorr_tools.select_random_templates)
# templates = random_templates.groupby(random_templates['OT'].dt.date).apply(lambda x: x.nlargest(4, 'N'))
# templates.index = templates.index.droplevel(level=[0, 1])
# print(templates)

events = templates[['lon', 'lat', 'depth', 'OT']]
min_lon = np.round(events['lon'].min()) - 1
max_lon = np.round(events['lon'].max()) + 1
min_lat = np.round(events['lat'].min()) - 1
max_lat = np.round(events['lat'].max()) + 1
day_of_month = templates['OT'].dt.day_of_year
year = templates['OT'].dt.year.unique()[0]
cmap = plt.cm.viridis

## Plot map for all data
plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([min_lon, max_lon, min_lat, max_lat])
ax.coastlines(resolution='10m', color='black', linewidth=1)
plt.scatter(events['lon'], events['lat'], marker='o', s=50, color='blue')
for i, txt in enumerate(range(1, len(events)+1)):
    plt.text(events['lon'].iloc[i] - 0.1, events['lat'].iloc[i], str(txt), fontsize=12, ha='left', va='center')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations of Events')
plt.grid(True)
plt.tight_layout()
plt.show()

## Plot 3D scatter plot

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
day_of_month = templates['OT'].dt.day_of_year
sc = ax.scatter(events['lon'], events['lat'], events['depth'], c=day_of_month, cmap=cmap, marker='o', s=10, label='Events')
cbar = plt.colorbar(sc)
cbar.set_label('Day of the Year', fontsize=14)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_zlabel('Depth', fontsize=14)
ax.set_xlim(-124.6,-122.9)#max_lon,min_lon)
ax.set_ylim(48,49.4)#min_lat,max_lat)
ax.set_zlim(events['depth'].min(), events['depth'].max())
ax.set_xticks(np.arange(max_lon,min_lon, 0.2))
ax.set_yticks(np.arange(min_lat,max_lat, 0.2))
plt.title(f"Year {year}: {len(templates)} detections", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.close()

## Plot 1 map for each month 

detections_per_month = templates.groupby(templates['OT'].dt.month).size()
for fig_index in range(3):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"Seismicity for Year {year}", fontsize=16)
    for sub_index, ax in enumerate(axes.flat):
        month_index = fig_index * 4 + sub_index + 1
        if month_index <= 12:
            month_data = templates[templates['OT'].dt.month == month_index]
            events_month = month_data[['lon', 'lat', 'depth', 'OT']]
            day_of_month = month_data['OT'].dt.dayofyear
            sc = ax.scatter(events_month['lon'], events_month['lat'], c=day_of_month, cmap=cmap, marker='o', s=10, label='Events')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Day of the Year', fontsize=10)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_xlim(max_lon, min_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.set_xticks(np.arange(max_lon, min_lon, 0.4))
            ax.set_yticks(np.arange(min_lat, max_lat, 0.4))
            ax.set_title(f"Month {month_index}\nDetections: {detections_per_month.get(month_index, 0)}")
            ax.grid(True)
            ax.coastlines()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(f'monthly_detect_2005_{fig_index+1}',dpi=300)
    plt.close()

## Latitude repartition by time

fig=plt.figure(figsize=(10, 8))
day_of_month = templates['OT'].dt.day_of_year
sc = plt.scatter(events['OT'], events['lat'], c=day_of_month, cmap=cmap, marker='o', s=10, label='Events')
cbar = plt.colorbar(sc)
cbar.set_label('Day of the Year', fontsize=14)
cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
ax.set_xlim(max_lon,min_lon)
ax.set_xticks(np.arange(max_lon,min_lon, 0.2))
plt.title(f"Year {year}: {len(templates)} detections", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.close()

## Daily rate

full_day_range = pd.date_range(start=startdate, end=enddate)
daily_rate = templates.groupby(templates['OT'].dt.dayofyear).size().reindex(full_day_range.dayofyear, fill_value=0)
first_day_of_month = pd.date_range(start=templates['OT'].iloc[0], end=templates['OT'].iloc[-1], freq='MS')
fig, ax = plt.subplots(figsize=(10, 6))
daily_rate.plot(kind='bar', ax=ax)
debut_mois = [0, 31, 61]
sse = [34, 55]
for day in debut_mois:
    ax.axvline(x=day, color='green', linestyle='--')
for day in sse:
    ax.axvline(x=day, color='red', linestyle='--')
plt.title('Daily Event Rate', fontsize=16)
plt.xlabel('Day of the Year', fontsize=14)
plt.ylabel('Number of Events', fontsize=14)
# ax.set_xticks(first_day_of_month.dayofyear)
# ax.set_xticklabels(first_day_of_month.strftime('%B'), rotation=45)
# plt.grid(True)
plt.tight_layout()
plt.show()

## Monthly rate

monthly_rate = templates.groupby(templates['OT'].dt.to_period('M')).size()
monthly_rate.index = monthly_rate.index.to_timestamp()
fig, ax = plt.subplots(figsize=(10, 6))
monthly_rate.plot(kind='bar', ax=ax)
plt.title('Monthly Event Rate', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Events', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.close()

## Percentage of daily rate values

bin_width=50
bins = range(0, max(daily_rate) + bin_width, bin_width)
hist, bin_edges = np.histogram(daily_rate, bins=bins)
fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=bin_width, align='edge')
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Event Rate")
total_values = len(daily_rate)
for i in range(len(hist)):
    percentage = hist[i] / total_values * 100
    plt.text(bin_edges[i], hist[i], f'{percentage:.1f}%', ha='center', va='bottom')
plt.close()

## Output

print("Number of events:", len(templates))
print("Period of time:", startdate.strftime("%Y-%m-%d"), "to", enddate.strftime("%Y-%m-%d"))
print("Range of depth:", events['depth'].min(), "to",events['depth'].max(),"km")
print("Range of events:", daily_rate.min(), "to", daily_rate.max() ,"events per day")
