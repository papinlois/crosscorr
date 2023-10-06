#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:02:46 2018

Autocorrelate

@author: amt
"""

###a voir pour une vraie sortie de cross correlation et non auto

import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from obspy.core import Stream, read, UTCDateTime
from obspy.signal.cross_correlation import correlate
import autocorr_tools
import pandas as pd
import matplotlib.colorbar as clrbar
from scipy import stats

#np.fft.restore_all()
startscript=time.time()

locfile=pd.read_csv('stations.csv')
locs=locfile.values

plt.figure()
plt.plot(locfile['Longitude'].tolist(),locfile['Latitude'].tolist(),'bo')
for ii in range(6):
    plt.text(locfile['Longitude'][ii],locfile['Latitude'][ii],locfile['Name'][ii])

st=Stream()

stas=['LZB','SNB']#,'NLLB','PGC']

mth=5
day=16
# Load data
for sta in stas:
    path = r"C:/Users/papin/Desktop/phd/data"
    file=(path+"\\"+'2010'+str(mth).zfill(2)+str(day).zfill(2)+'.CN.'+sta+'..BHE.mseed')

    # if net=='PB' or net=='UW':
    #     filename = (path + "\\"+ sta + '.' + net + '.' +
    #                 yr + '.' + day)
    # elif net=='CN' or net=='NTKA':
    #     filename = (path + "\\" + yr+mth+tod+'.'+net+'.'+sta+'..'+cha+'.mseed')
    st += read(file)
    print(file)
    
# Clip to same time
start=st[0].stats.starttime
end=st[0].stats.endtime
for ii in range(1,len(st)):
    if start<st[ii].stats.starttime:
        start=st[ii].stats.starttime
    if end>st[ii].stats.endtime:
        end=st[ii].stats.endtime
st.interpolate(sampling_rate=80, starttime=start)
st.trim(starttime=start, endtime=end,nearest_sample=1,pad=1,fill_value=0)

# Filterdata  
st.detrend(type='simple')
st.filter("bandpass", freqmin=1.0, freqmax=10.0)
dt=timedelta(minutes=60)
hr=12
st=st.trim(start+hr*dt,start+hr*dt+dt)

# Add locations
for ii in range(0,len(stas)):       
    ind=np.where(locs[:,0]==stas[ii])  
    print(locs[ind,0][0][0])
    st[ii].stats.y=locs[ind,1][0][0]
    st[ii].stats.x=locs[ind,2][0][0]

## Plot data
plt.figure()
for tmp in range(len(st)):
    plt.plot(st[tmp].times("timestamp"),st[tmp].data/np.max(np.abs(st[tmp].data))+tmp*1,color='tab:blue')
#plt.ylim((-200,1000))
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Normalized Data + Offset', fontsize=14)
plt.legend(stas, loc='upper right', fontsize=12)
plt.grid(True)

# Cross correlate
windowdur=6 # template window duration in seconds
windowlen=int(windowdur*st[0].stats.sampling_rate) # template window length in pts
windowstep=3 # time shift for next window in seconds
windowsteplen=int(windowstep*st[0].stats.sampling_rate) # time shift in pts
numwindows=int((st[0].stats.npts-windowlen)/windowsteplen) # number of time windows in interval
leng=len(st[0].data)
xcorrmean=np.zeros((numwindows,st[0].stats.npts-windowlen+1))

##########################

# Autocorrelate
for ii in range(len(st)):
    startcode = time.time()
    xcorrfull=np.zeros((numwindows,st[0].stats.npts-windowlen+1))
    for kk in range(numwindows):          
        xcorrfull[kk,:]=autocorr_tools.correlate_template(st[ii].data, st[ii].data[(kk*windowsteplen):(kk*windowsteplen+windowlen)], 
                mode='valid', normalize='full', demean=True, method='auto')
        # print(kk)
    xcorrmean+=xcorrfull
    endcode = time.time()
    print(endcode-startcode)

# Network autocorrelation
xcorrmean=xcorrmean/len(st)

## Save output
#hf = h5py.File('xcorrfullsummed.h5', 'w')
#hf.create_dataset('dataset_1', data=xcorrmean)
#hf.close()

# Median absolute deviation
mad=np.median(np.abs(xcorrmean - np.median(xcorrmean))) #Median absolute deviation
thresh=8
aboves=np.where(xcorrmean>thresh*mad)

# Detection plot
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(aboves[0],aboves[1],s=20,c=xcorrmean[aboves])
ax.set_xlabel('Template Index', fontsize=14)
ax.set_ylabel('Time Index', fontsize=14)
cax, _ = clrbar.make_axes(ax)
cbar = clrbar.ColorbarBase(cax)
cbar.ax.set_ylabel('Correlation Coefficient', rotation=270, labelpad=15, fontsize=14)
ax.set_xlim((np.min(aboves[0]),np.max(aboves[0])))
ax.set_ylim((np.min(aboves[1]),np.max(aboves[1])))

# 
winind=stats.mode(aboves[0])[0][0]
xcorr=xcorrmean[winind,:]
fig, ax = plt.subplots(figsize=(10,3))
t=st[0].stats.delta*np.arange(len(xcorr))
ax.plot(t,xcorr)
ax.axhline(thresh*mad,color='red')
inds=np.where(xcorr>thresh*mad)[0]
clusters=autocorr_tools.clusterdects(inds,windowlen)
newdect=autocorr_tools.culldects(inds,clusters,xcorr)
ax.plot(newdect*st[0].stats.delta,xcorr[newdect],'kx')
ax.text(60,1.1*thresh*mad,'8*MAD',fontsize=16,color='red')
ax.set_xlabel('Seconds of Hour 12 on 18/5', fontsize=14)
ax.set_ylabel('Correlation Coefficient', fontsize=14) ###
ax.set_xlim((0,3600))
plt.gcf().subplots_adjust(bottom=0.2)

if len(newdect)>1:
    for staind in range(len(st)):
        fig,ax=plt.subplots(figsize=(8,10))
        t=st[staind].stats.delta*np.arange(windowlen+4*windowsteplen) ####
        snippet=st[staind].data[(winind*windowsteplen-3*windowsteplen):(winind*windowsteplen+windowlen+1*windowsteplen)] #####
        plt.plot(t,snippet/np.max(np.abs(snippet)),color=(0.4,0.4,0.4))
        plt.text(-0.6,0.5,'Template')
        stack=np.zeros(windowlen+4*windowsteplen) ####
        for ii in range(len(newdect)):
            if newdect[ii] != winind:
                ind=int(newdect[ii])
                startsnip=ind-3*windowsteplen ####
                stopsnip=ind+windowlen+1*windowsteplen ####
                wf=st[staind].data[startsnip:stopsnip]
                if ii < 50:
                    plt.plot(t,wf/np.max(np.abs(wf))-ii-1,color=(0.75,0.75,0.75))
                    plt.text(-0.6,-ii-1,'cc='+str(int(100*xcorr[ind])/100))
                stack+=wf/np.max(np.abs(wf))
        plt.plot(t,stack/np.max(np.abs(stack))+2)
        plt.text(-0.6,2.5,'Stack')
        plt.xlim((np.min(t),np.max(t)))
        plt.ylim((-ii-2,3.5))
        ax.spines['top'].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Time (s)',fontsize=16)
        plt.title(st[staind].stats.station+'-'+st[staind].stats.channel)

endscript=time.time()
print(endscript-startscript)