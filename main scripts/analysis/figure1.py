#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:59:46 2020

Make figure 1 data

@author: amt
"""
'''
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.gridspec as gridspec
import h5py
import numpy as np

sr=100

# LOAD THE DATA
print("LOADING DATA")
x_data = h5py.File('data/h5/ID_256_PO.TWKB.HH_S.h5', 'r')

fig1 = plt.figure(constrained_layout=True,figsize=(10,10))
gs = fig1.add_gridspec(3, 1)
f1_ax1 = fig1.add_subplot(gs[0:2, 0])

# make stack
stack = np.zeros(x_data['waves'].shape[1])
for ii in range(x_data['waves'].shape[0]):
    # plt.figure()
    # plt.plot(x_data['waves'][ii,:])
    stack+=x_data['waves'][ii,:]/np.max(np.abs(x_data['waves'][ii,:]))

# plot template and detection relative to origin time
np.random.seed(8)
inds=np.random.choice(x_data['waves'].shape[0],20)
print(inds)
t=1/sr*np.arange(x_data['waves'].shape[1])
for ii,ind in enumerate(inds):
    print(ii)
    clip=x_data['waves'][ind,:]
    f1_ax1.plot(t,clip/np.max(1*np.abs(clip))+ii,color=(0.6,0.6,0.6))
f1_ax1.plot(t,stack/np.max(stack)+ii+1,color=(0.3,0.3,0.3))

f1_ax1.set_xlim((0,90))
f1_ax1.set_ylim((-2,21))
f1_ax1.set_yticks([], [])
f1_ax1.text(1,-1.5,'A',fontsize=28,fontweight='bold')
f1_ax1.tick_params(axis="x", labelsize=12)
f1_ax1.axvline(x=30,c='k')
f1_ax1.axvline(x=60,c='k')
f1_ax1.text(1,20.25,'stack',fontsize=12,color=(0.3,0.3,0.3))
#f1_ax1.text(13,-1.55,'DP1',fontsize=16,color=(0,0,0),fontweight="bold")
#f1_ax1.text(43,-1.55,'DP2',fontsize=16,color=(0,0,0),fontweight="bold")
#f1_ax1.text(73,-1.55,'DP3',fontsize=16,color=(0,0,0),fontweight="bold")

f1_ax2 = fig1.add_subplot(gs[2, 0])
f1_ax2.set_xlabel('Time (s)',fontsize=14)
f1_ax2.set_xlim((0,30))

t=1/sr*np.arange(x_data['waves'].shape[1]//3)
clip=stack
f1_ax2b = f1_ax2.twinx() 
f1_ax2b.set_yticks([], [])

#f1_ax2.plot(t,signal.gaussian(len(clip)//3,std=int(sr*0.05)),label='$\sigma$=0.05 s',color=(0.5,0.,0.), lw=2)
#f1_ax2.plot(t,signal.gaussian(len(clip)//3,std=int(sr*0.1)),label='$\sigma$=0.1 s',color=(0.,0.5,0.), lw=2)
#f1_ax2.plot(t,signal.gaussian(len(clip)//3,std=int(sr*0.2)),label='$\sigma$=0.2 s',color=(0.,0.,0.5), lw=2)
#f1_ax2.plot(t,signal.gaussian(len(clip)//3,std=int(sr*0.4)),label='$\sigma$=0.4 s',color=(0.5,0.5,0.), lw=2)
f1_ax2.set_ylim((-0.02,1.02))
#legend=f1_ax2.legend(framealpha=1,edgecolor="black",frameon=True,prop={'size': 12},ncol=2, bbox_to_anchor=(0.0161,0.0163,0.15,0.15)) #.zorder(20)
f1_ax2.set_ylabel("Target Amplitude",fontsize=14)
#legend.remove()
#f1_ax2.add_artist(legend)
#frame = legend.get_frame()
#frame.set_facecolor('white')
#frame.set_edgecolor('black')

clip1=clip[:len(clip)//3]
clip2=clip[len(clip)//3:2*len(clip)//3]
clip3=clip[2*len(clip)//3:]
f1_ax2b.plot(t,clip1/np.max(1*np.abs(clip1))+2,color=(0.3,0.3,0.3), lw=1.5)
f1_ax2b.plot(t,clip2/np.max(1*np.abs(clip2))+1,color=(0.3,0.3,0.3), lw=1.5)
f1_ax2b.plot(t,clip3/np.max(1*np.abs(clip3))-0,color=(0.3,0.3,0.3), lw=1.5)
#f1_ax2b.text(10.25,2.2,'DP1 SMNB stack',fontsize=14,color=(0.3,0.3,0.3))
#f1_ax2b.text(10.25,1.2,'DP2 SMNB stack',fontsize=14,color=(0.3,0.3,0.3))
#f1_ax2b.text(10.25,.2,'DP3 SMNB stack',fontsize=14,color=(0.3,0.3,0.3))
f1_ax2b.set_ylim((-1.1,3.1))
f1_ax2.set_xlim((5,25))
f1_ax2.tick_params(axis="x", labelsize=12)
f1_ax2.tick_params(axis="y", labelsize=12)
f1_ax2.text(10.1,0.9,'B',fontsize=28,fontweight='bold')


# legend.get_frame().set_alpha(1)
# legend.get_frame().set_facecolor((0, 0, 1, 0.1))

# leg = ax.legend()
# leg.remove()
# ax2.add_artist(leg)
    
fig1.savefig("figure1.png")

'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:59:46 2020

Make figure 1 data

@author: amt
"""

## Only the plot of the stack

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import numpy as np

sr=100

# LOAD THE DATA
print("LOADING DATA")
x_data = h5py.File('data/h5/ID_002_PO.SSIB.HH_S.h5', 'r')

# make stack
stack = np.zeros(x_data['waves'].shape[1])
random_indices = np.random.choice(x_data['waves'].shape[0], 50, replace=False)  # Choose 50 random indices without replacement
for ii in random_indices:  # Iterate over the randomly chosen indices
    stack += x_data['waves'][ii, :] / np.max(np.abs(x_data['waves'][ii, :]))

# plot template and detection relative to origin time
fig1 = plt.figure(constrained_layout=True, figsize=(10, 5))
f1_ax2 = fig1.add_subplot(111)

f1_ax2.set_xlabel('Time (s)', fontsize=14)
f1_ax2.set_xlim((0, 30))

t = 1/sr * np.arange(x_data['waves'].shape[1]//3)
clip = stack
f1_ax2b = f1_ax2.twinx()
f1_ax2b.set_yticks([], [])

clip1 = clip[:len(clip)//3]
clip2 = clip[len(clip)//3:2*len(clip)//3]
clip3 = clip[2*len(clip)//3:]
f1_ax2b.plot(t, clip1/np.max(1*np.abs(clip1)) + 2, color=(0.3, 0.3, 0.3), lw=1.5)
f1_ax2b.plot(t, clip2/np.max(1*np.abs(clip2)) + 1, color=(0.3, 0.3, 0.3), lw=1.5)
f1_ax2b.plot(t, clip3/np.max(1*np.abs(clip3)) - 0, color=(0.3, 0.3, 0.3), lw=1.5)

f1_ax2b.set_ylim((-1.1, 3.1))
f1_ax2.set_xlim((5, 25))
f1_ax2.tick_params(axis="x", labelsize=12)
f1_ax2.tick_params(axis="y", labelsize=12)
f1_ax2.text(10.1, 0.9, 'B', fontsize=28, fontweight='bold')

fig1.savefig("figure2.png")