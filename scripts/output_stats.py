#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:18:28 2024

Compute some maths on the output file full of the new events.

@author: lpapin

As of 17/06/24.
"""

## Detections
import pandas as pd
detections=pd.read_csv('/Users/lpapin/Desktop/SSE_2005/crosscorr/aug_PO_2/yes/output_aug_PO_yes.txt')
# detections=detections.sort_values(['starttime'])
detections.reset_index(inplace=True, drop=False)
detections.index.name = 'Index'
print(detections)
# Filtering only "high" cc value
# coeff_thresh=0.2
# filtered_detections = detections[detections['coeff'] > coeff_thresh]
# print(filtered_detections)
# detections=filtered_detections

## Plotting the events timeline
import matplotlib.pyplot as plt
import numpy as np
detections['starttime'] = pd.to_datetime(detections['starttime'])
templates = detections['template'].unique()
colors = plt.cm.get_cmap('tab20', len(templates))
templ_colors = {template: colors(i) for i, template in enumerate(templates)}
detections['color'] = detections['template'].map(templ_colors)
detections['size'] = np.log(detections['coeff'] + 1) * 200
plt.figure(figsize=(12, 5))
sc = plt.scatter(detections['starttime'], detections.index, c=detections['color'], 
                 s=detections['size'], marker='o')
for template, color in templ_colors.items():
    max_coeff_idx = detections[detections['template'] == template]['coeff'].idxmax()
    max_coeff_date = detections.loc[max_coeff_idx, 'starttime']
    plt.scatter(max_coeff_date, max_coeff_idx, color=color, marker='*', edgecolor='black', s=200)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=template, 
                      markersize=10, markerfacecolor=color) 
           for template, color in templ_colors.items()]
plt.legend(handles=handles, title='Template', bbox_to_anchor=(1, 1), loc='upper left')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Number of Detections', fontsize=14)
plt.grid(True)
# plt.xlim(pd.Timestamp("2005-08-26"), pd.Timestamp("2005-08-27"))
plt.savefig('events_timeline.png', dpi=300)
plt.tight_layout()
plt.show()

## Plot number of detections per day, stacked by template
detections_templ = detections['template'].value_counts().sort_index()
detections['date'] = detections['starttime'].dt.date
detections_day = detections.groupby(['date', 'template']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 5))
colors = plt.cm.get_cmap('tab20', len(detections_day.columns))
for i, template in enumerate(detections_day.columns):
    plt.bar(detections_day.index, detections_day[template], bottom=detections_day.iloc[:, :i].sum(axis=1),
            color=colors(i), label=template)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Detections', fontsize=14)
plt.title('Number of Detections per Day (Stacked by Template)', fontsize=16)
plt.legend(title='Template', bbox_to_anchor=(1, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y')
for template, color in zip(detections_day.columns, colors.colors):
    max_coeff_date = detections[detections['template'] == template]['starttime'].dt.date[
        detections[detections['template'] == template]['coeff'].idxmax()]
    max_value = detections_day.loc[max_coeff_date, template]
    plt.annotate('*', xy=(max_coeff_date, max_value), xytext=(0, 5), textcoords='offset points',
                 color=color, fontsize=24, ha='center', va='bottom',
                 bbox=dict(facecolor='none', edgecolor='none', pad=0.5))
plt.tight_layout()
plt.savefig('detections_day.png', dpi=300)
plt.show()

## Plotting the events cc values
bins = np.arange(0, 1.1, 0.1)
detections['coeff_bin'] = pd.cut(detections['coeff'], bins, right=False)
detections_bin_templ = detections.groupby(['coeff_bin', 'template'], observed=True).size().unstack(fill_value=0)
detections_bin_templ.plot(kind='bar', stacked=True, figsize=(12, 5), colormap='tab20')
plt.xlabel('Coeff', fontsize=14)
plt.ylabel('Number of Detections', fontsize=14)
plt.title('Number of Detections per Crosscorr Coeff (Stacked by Template)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Template', bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('coeff_cc', dpi=300)
plt.show()

## 
mean_coeff_templ = detections.groupby('template')['coeff'].mean()
print("\nMean Coefficient per Template:")
print(mean_coeff_templ)
mean_coeff_day = detections.groupby('date')['coeff'].mean()
print("\nMean Coefficient per Day:")
print(mean_coeff_day)


