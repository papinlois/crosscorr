#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:15:44 2018

Autocorrelation tools

@author: amt
"""

from itertools import groupby, count
import numpy as np
from sklearn.cluster import DBSCAN
import scipy
import scipy.signal
from distutils.version import LooseVersion
from obspy import Stream, Trace
import matplotlib.pyplot as plt
from scipy.signal import hilbert, correlate
from obspy.signal.trigger import classic_sta_lta
from sklearn.neural_network import MLPClassifier
import pickle

def makexcorrs(st,numwindows,windowlen,windowsteplen):
    xcorrfull=np.zeros((numwindows,st.stats.npts-windowlen+1))
    for kk in range(numwindows):          
        xcorrfull[kk,:]=correlate_template(st.data, st.data[(kk*windowsteplen):(kk*windowsteplen+windowlen)], 
                 mode='valid', normalize='full', demean=True, method='auto') 
    return xcorrfull

def xcorr_detection_stacks_plot(st,repeats,newdect,xcorrtot):
    '''
    Calculates xcorr detection stacks and SNRs 
    Returns normalized stacks and SNRs
    Same as xcorr_detection_stacks but returns plots also
    '''
    frontpad=3 # seconds
    backpad=5 # seconds
    sr=st[0].stats.sampling_rate
    delta=st[0].stats.delta
    frontpadn=int(frontpad*sr)
    backpadn=int(backpad*sr)
    stacks=np.empty((len(st),frontpadn+backpadn))
    pwsstacks=np.empty((len(st),frontpadn+backpadn))
    stacksnrs=np.empty(len(st))
    pwsstacksnrs=np.empty(len(st))
    t=delta*np.arange(frontpadn+backpadn)
    for staind in range(len(st)):
        fig,ax=plt.subplots(figsize=(8,10))
        # plot template waveform
        stack=np.zeros(len(t)) ####
        repeaterwaveforms=np.zeros((repeats,len(t)))
        for ii in range(repeats):
            ind=int(newdect[ii])
            if ind-frontpadn < 0:
                wf=np.concatenate((np.zeros(np.abs(ind-frontpadn)),st[staind].data[np.arange(0,ind+backpadn)]))
            elif ind+backpadn > st[0].stats.npts:
                wf=np.concatenate((st[staind].data[(ind-frontpadn):st[0].stats.npts],np.zeros(ind+backpadn-st[0].stats.npts)))
            else:
                wf=st[staind].data[np.arange(ind-frontpadn,ind+backpadn)]
            if ii < 50:
                plt.plot(t,wf/np.max(np.abs(wf))-ii-1,color=(0.75,0.75,0.75))
                plt.text(-0.6,-ii-1,'cc='+str(int(100*xcorrtot[ii])/100))
            repeaterwaveforms[ii,:]=wf
            stack+=wf/np.max(np.abs(wf))
        stacks[staind,:]=stack
        pws=PWS_stack(repeaterwaveforms)
        pwsstacks[staind,:]=pws/np.max(np.abs(pws))
        stacksnr=str(np.max(np.abs(stack))/np.median(np.abs(stack)))
        pwsstacksnr=str(np.max(np.abs(pws))/np.median(np.abs(pws)))
        stacksnrs[staind]=stacksnr
        pwsstacksnrs[staind]=pwsstacksnr
        plt.plot(t,stack/np.max(np.abs(stack))+2)
        plt.plot(t,pws/np.max(np.abs(pws))+2)
        plt.text(-0.6,2.5,'Stack-'+stacksnr[:5])
        plt.xlim((np.min(t),np.max(t)))
        ax.spines['top'].set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Time (s)',fontsize=16)
        plt.title(st[staind].stats.station+'-'+st[staind].stats.channel)
    return stacks, pwsstacks, stacksnrs, pwsstacksnrs

def xcorr_detection_stacks(st,repeats,newdect,xcorrtot):
    '''
    Calculates xcorr detection stacks and SNRs 
    Returns normalized stacks and SNRs
    '''
    frontpad=3 # seconds
    backpad=5 # seconds
    sr=st[0].stats.sampling_rate
    delta=st[0].stats.delta
    frontpadn=int(frontpad*sr)
    backpadn=int(backpad*sr)
    stacks=np.empty((len(st),frontpadn+backpadn))
    pwsstacks=np.empty((len(st),frontpadn+backpadn))
    stacksnrs=np.empty(len(st))
    pwsstacksnrs=np.empty(len(st))
    t=delta*np.arange(frontpadn+backpadn)
    for staind in range(len(st)):
        #fig,ax=plt.subplots(figsize=(8,10))
        # plot template waveform
        stack=np.zeros(len(t)) ####
        repeaterwaveforms=np.zeros((repeats,len(t)))
        for ii in range(repeats):
            ind=int(newdect[ii])
            if ind-frontpadn < 0:
                wf=np.concatenate((np.zeros(np.abs(ind-frontpadn)),st[staind].data[np.arange(0,ind+backpadn)]))
            elif ind+backpadn > st[0].stats.npts:
                wf=np.concatenate((st[staind].data[(ind-frontpadn):st[0].stats.npts],np.zeros(ind+backpadn-st[0].stats.npts)))
            else:
                wf=st[staind].data[np.arange(ind-frontpadn,ind+backpadn)]
#            if ii < 50:
                #plt.plot(t,wf/np.max(np.abs(wf))-ii-1,color=(0.75,0.75,0.75))
                #plt.text(-0.6,-ii-1,'cc='+str(int(100*xcorrtot[ii])/100))
            repeaterwaveforms[ii,:]=wf
            stack+=wf/np.max(np.abs(wf))
        stacks[staind,:]=stack
#        pws=PWS_stack(repeaterwaveforms)
#        pwsstacks[staind,:]=pws/np.max(np.abs(pws))
        stacksnr=str(np.max(np.abs(stack))/np.median(np.abs(stack)))
#        pwsstacksnr=str(np.max(np.abs(pws))/np.median(np.abs(pws)))
        stacksnrs[staind]=stacksnr
#        pwsstacksnrs[staind]=pwsstacksnr
#        plt.plot(t,stack/np.max(np.abs(stack))+2)
#        plt.plot(t,pws/np.max(np.abs(pws))+2)
#        plt.text(-0.6,2.5,'Stack-'+stacksnr[:5])
#        plt.xlim((np.min(t),np.max(t)))
#        ax.spines['top'].set_visible(False)
#        ax.axes.get_yaxis().set_visible(False)
#        ax.spines['left'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        plt.xlabel('Time (s)',fontsize=16)
#        plt.title(st[staind].stats.station+'-'+st[staind].stats.channel)
    return stacks, stacksnrs

def plot_stacks(stacks,sr,pwssnrs):
    fig,ax=plt.subplots(figsize=(12,10))
    t=1/sr*np.arange(np.shape(stacks)[1])
    for staind in range(np.shape(stacks)[0]):
        #env=np.abs(hilbert(stacks[staind,:]/np.max(np.abs(stacks[staind,:]))))
        ax.plot(t,stacks[staind,:]/np.max(np.abs(stacks[staind,:]))-staind)
        ax.text(t[0]-0.5,-staind,str(np.round(pwssnrs[staind])))  
    ax.spines['top'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
def stalta_picker(stacks,sr,stations,snrs): 
    picks=np.zeros(len(stacks))
    tmp=np.zeros(len(stacks[0,:]))
    for ii in range(len(stacks)):
        tmp+=stacks[ii,:]/np.max(np.abs(stacks[ii,:]))
    maxind=np.where(tmp==np.max(tmp))[0][0]
    if len(np.where(snrs>20)[0]) > 4:        
        swin=0.05*sr # len in samples
        lwin=0.25*sr
        t=1/sr*np.arange(len(tmp))
        for staind in range(np.shape(stacks)[0]):
            if snrs[staind] > 20:
                #print(snrs[staind])
                cf=classic_sta_lta(stacks[staind,:],swin,lwin)
                #cf=cf[(maxind-nwin//2):(maxind+nwin//2)]
                maxi=np.max(cf)
                maxiloc=np.where(cf==maxi)[0][0]
                thresh=np.max(np.abs(cf))/np.median(np.abs(cf))
                #print(thresh)
#                print(maxind)
                aboves=np.where(cf>thresh/2)[0] # list of values above triggering threshold
                withins=np.arange(maxind-35,maxind+20) # list of values withing 50 samples of max
                aboves=np.intersect1d(aboves,withins)
                if len(aboves)>0:
                    picks[staind]=t[aboves[0]]
                else:
                    picks[staind]=0
            else:
                picks[staind]=0
            # plot picks
#        if np.sum(picks)>0:
#            plt.figure(figsize=(6,10))
#            for ii in range(len(stacks)):
#                if picks[ii] != 0:
#                    plt.vlines(picks[ii],-stations[ii]-0.5,-stations[ii]+0.5)
#                plt.plot(t,stacks[ii,:]/np.max(stacks[ii,:])-stations[ii])
#                plt.text(2.0,-stations[ii],str(np.round(snrs[ii])))
#                #plt.text(2.0,-master[templateid]['stations'][ii],str(np.round(master[templateid]['snrs'][ii])))              
#            plt.vlines(t[maxind],-np.max(stations)-0.5,0.5)
    return picks

def mlp_picker(stacks,sr,snrs): 
    # load autopicker
    with open('mlp_picker_10_10.pkl', 'rb') as f:
        mlp = pickle.load(f)
    picks=np.zeros(len(stacks))
    if len(np.where(snrs>20)[0]) > 4:
        t=1/sr*np.arange(len(stacks[0,:]))
        normstack=np.zeros_like(stacks)
        for staind in range(np.shape(stacks)[0]):
            normstack[staind,:]=stacks[staind,:]/np.max(np.abs(stacks[staind,:]))
        picks = mlp.predict(normstack)
        for staind in range(np.shape(stacks)[0]):
            if snrs[staind] < 20:
                picks[staind] = 0
#            plt.plot(t,normstack[staind])
#            plt.axvline(x=t[picks],color=(0,0,0))
        print(picks)
        picks=picks/sr
        print(picks)
    return picks

def xcorr_envelopes(stacks,snrs,stations,refstaind):    
    ref=np.abs(hilbert(stacks[refstaind,:]))
    plt.figure(figsize=(12,10))
    print(stations[refstaind])
    ax1=plt.subplot(121)
    plt.plot(ref/np.max(np.abs(ref))-stations[refstaind],color=(0,0,0),linestyle=':') # reference station envelope
    plt.plot(stacks[refstaind,:]/np.max(np.abs(stacks[refstaind,:]))-stations[refstaind],color=(0,0,0)) # reference station waveform
    ax2=plt.subplot(122,sharex=ax1, sharey=ax1)
    plt.plot(ref/np.max(np.abs(ref))-stations[refstaind],color=(0,0,0),linestyle=':') # reference station envelope
    plt.plot(stacks[refstaind,:]/np.max(np.abs(stacks[refstaind,:]))-stations[refstaind],color=(0,0,0)) # reference station waveform
    outstacks=np.empty(np.shape(stacks))
    for staind in np.where(snrs>20)[0]: #range(np.shape(stacks)[0]):
        if staind != refstaind:
            ax1=plt.subplot(121)
            plt.plot(stacks[staind,:]/np.max(np.abs(stacks[staind,:]))-stations[staind])
            b_sig=np.abs(hilbert(stacks[staind,:]))
            b_sig=b_sig/np.max(np.abs(b_sig))
            plt.plot(b_sig-stations[staind],linestyle=':')
            lag = np.argmax(correlate(ref, b_sig))
            #lag = np.argmax(correlate(stacks[refstaind,:]/np.max(np.abs(stacks[refstaind,:])), stacks[staind,:]/np.max(np.abs(stacks[staind,:]))))
            print(lag)
        else: 
            lag=0
        outstacks[staind,:] = np.roll(stacks[staind,:], shift=int(np.ceil(lag)))
        ax2=plt.subplot(122,sharex=ax1, sharey=ax1)
        plt.plot(np.roll(b_sig, shift=int(np.ceil(lag)))-stations[staind],linestyle=':')
        plt.plot(np.roll(stacks[staind,:]/np.max(np.abs(stacks[staind,:])), shift=int(np.ceil(lag)))-stations[staind])
    return outstacks 

def PWS_stack(streams,Normalize=1):
    """
    Compute the phase weighted stack of a series of streams.
    Assume streams are already

    """
    # First get the linear stack which we will weight by the phase stack
    if Normalize:
        maxs=np.max(np.abs(streams),axis=1)
        streams=streams/maxs[:,None]
    Linstack = streams #np.sum(streams,0) 
    phas=np.zeros_like(streams)
    for ii in range(np.shape(phas)[0]):
        tmp=hilbert(streams[ii,:]) # hilbert transform of each timeseries
        phas[ii,:]=np.arctan2(np.imag(tmp),np.real(tmp)) # instantaneous phase using the hilbert transform 
    sump=np.abs(np.sum(np.exp(np.complex(0,1)*phas),axis=0))/np.shape(phas)[0]  
    Phasestack=sump*Linstack # traditional stack*phase stack 
#    hilb=hilbert(seis').'; % hilbert transform of each timeseries
#    phas=atan2(imag(hilb),real(hilb)); % instantaneous phase using the hilbert transform
#    sump=real(abs(mean(exp(sqrt(-1)*phas)))).^1;
#    sums=mean(seis); % traditional stack
#    stack=sump.*sums; % traditional stack*phase stack    
    return Phasestack

def generate_detection_plot(st_filt,newdect,xcorrtot):
    """
    Plot detection waveforms
    Inputs:
        st_filt=day long stream
        newdect=declustered detection indicies corresponding to st_filt
        xcorrtot=newtork cross correlations for window
    """
    frontpad=3 # seconds
    backpad=5 # seconds
    sr=st_filt[0].stats.sampling_rate
    delta=st_filt[0].stats.delta
    frontpadn=int(frontpad*sr)
    backpadn=int(backpad*sr)
    if len(newdect)>1:
        for staind in range(len(st_filt)):
            fig,ax=plt.subplots(figsize=(8,10))
            snipstart=np.where(xcorrtot==np.max(xcorrtot))[0][0]
            if snipstart-frontpadn < 0:
                snippet=np.concatenate((np.zeros(np.abs(snipstart-frontpadn)),st_filt[staind].data[np.arange(0,snipstart+backpadn)]))
            else:
                snippet=st_filt[staind].data[np.arange(snipstart-frontpadn,snipstart+backpadn)]
            t=delta*np.arange(len(snippet))
            plt.plot(t,snippet/np.max(np.abs(snippet)),color=(0.4,0.4,0.4))
            plt.text(-0.6,0.5,'Template')
            stack=np.zeros(len(snippet)) ####
            for ii in range(len(newdect)):
                ind=int(newdect[ii])
                if ind-frontpadn < 0:
                    wf=np.concatenate((np.zeros(np.abs(snipstart-frontpadn)),st_filt[staind].data[np.arange(0,ind+backpadn)]))
                elif ind+backpadn > st_filt[0].stats.npts:
                    wf=np.concatenate((st_filt[staind].data[(ind-frontpadn):st_filt[0].stats.npts],np.zeros(ind+backpadn-st_filt[0].stats.npts)))
                else:
                    wf=st_filt[staind].data[np.arange(ind-frontpadn,ind+backpadn)]
                if ii < 50:
                    plt.plot(t,wf/np.max(np.abs(wf))-ii-1,color=(0.75,0.75,0.75))
                    plt.text(-0.6,-ii-1,'cc='+str(int(100*xcorrtot[ind])/100))
                stack+=wf/np.max(np.abs(wf))
            plt.plot(t,stack/np.max(np.abs(stack))+2)
            plt.text(-0.6,2.5,'Stack')
            plt.xlim((np.min(t),np.max(t)))
            ax.spines['top'].set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel('Time (s)',fontsize=16)
            plt.title(st_filt[staind].stats.station+'-'+st_filt[staind].stats.channel)

def clusterdects(dects,windowlen):
    dbscan_dataset1 = DBSCAN(eps=windowlen, min_samples=1, metric='euclidean').fit_predict(dects.reshape(-1, 1))
    dbscan_labels1 = dbscan_dataset1
    return dbscan_labels1

def culldects(dects,clusters,xcorr):
    newdect=np.empty(clusters[-1]+1,dtype=int)
    for ii in range(clusters[-1]+1):
#        print('ii='+str(ii))
        tinds=np.where(clusters==ii)[0]
#        print(tinds)
        dectinds=dects[tinds]
#        print(dectinds)
        values=xcorr[dectinds]
#        print(values)
#        print(np.argmax(values))
#        print(dects[np.argmax(values)])
        newdect[ii]=int(dectinds[np.argmax(values)])     
    return newdect

def _window_sum(data, window_len):
    """Rolling sum of data"""
    window_sum = np.cumsum(data)
    # in-place equivalent of
    # window_sum = window_sum[window_len:] - window_sum[:-window_len]
    # return window_sum
    np.subtract(window_sum[window_len:], window_sum[:-window_len],
                out=window_sum[:-window_len])
    return window_sum[:-window_len]

def _pad_zeros(a, num, num2=None):
    """Pad num zeros at both sides of array a"""
    if num2 is None:
        num2 = num
    hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
    return np.hstack(hstack)

def _call_scipy_correlate(a, b, mode, method):
    """
    Call the correct correlate function depending on Scipy version and method
    """
    if LooseVersion(scipy.__version__) >= LooseVersion('0.19'):
        cc = scipy.signal.correlate(a, b, mode=mode, method=method)
    elif method in ('fft', 'auto'):
        cc = scipy.signal.fftconvolve(a, b[::-1], mode=mode)
    elif method == 'direct':
        cc = scipy.signal.correlate(a, b, mode=mode)
    else:
        msg = "method keyword has to be one of ('auto', 'fft', 'direct')"
        raise ValueError(msg)
    return cc

def correlate_template(data, template, mode='valid', normalize='full',
                       demean=True, method='auto'):
    """
    Normalized cross-correlation of two signals with specified mode.

    If you are interested only in a part of the cross-correlation function
    around zero shift consider using function
    :func:`~obspy.signal.cross_correlation.correlate` which allows to
    explicetly specify the maximum shift.

    :type data: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param data: first signal
    :type template: :class:`~numpy.ndarray`, :class:`~obspy.core.trace.Trace`
    :param template: second signal to correlate with first signal.
        Its length must be smaller or equal to the length of ``data``.
    :param str mode: correlation mode to use.
        It is passed to the used correlation function.
        See :func:`scipy.signal.correlate` for possible options.
        The parameter determines the length of the correlation function.
    :param normalize:
        One of ``'naive'``, ``'full'`` or ``None``.
        ``'full'`` normalizes every correlation properly,
        whereas ``'naive'`` normalizes by the overall standard deviations.
        ``None`` does not normalize.
    :param demean: Demean data beforehand. For ``normalize='full'`` data is
        demeaned in different windows for each correlation value.
    :param str method: Method to use to calculate the correlation.
         ``'direct'``: The correlation is determined directly from sums,
         the definition of correlation.
         ``'fft'`` The Fast Fourier Transform is used to perform the
         correlation more quickly.
         ``'auto'`` Automatically chooses direct or Fourier method based on an
         estimate of which is faster. (Only availlable for SciPy versions >=
         0.19. For older Scipy version method defaults to ``'fft'``.)

    :return: cross-correlation function.

    .. note::
        Calling the function with ``demean=True, normalize='full'`` (default)
        returns the zero-normalized cross-correlation function.
        Calling the function with ``demean=False, normalize='full'``
        returns the normalized cross-correlation function.

    .. rubric:: Example

    >>> from obspy import read
    >>> data = read()[0]
    >>> template = data[450:550]
    >>> cc = correlate_template(data, template)
    >>> index = np.argmax(cc)
    >>> index
    450
    >>> round(cc[index], 9)
    1.0
    """
    # if we get Trace objects, use their data arrays
    if isinstance(data, Trace):
        data = data.data
    if isinstance(template, Trace):
        template = template.data
    data = np.asarray(data)
    template = np.asarray(template)
    lent = len(template)
    if len(data) < lent:
        raise ValueError('Data must not be shorter than template.')
    if demean:
        template = template - np.mean(template)
        if normalize != 'full':
            data = data - np.mean(data)
    cc = _call_scipy_correlate(data, template, mode, method)
    if normalize is not None:
        tnorm = np.sum(template ** 2)
        if normalize == 'naive':
            norm = (tnorm * np.sum(data ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize == 'full':
            pad = len(cc) - len(data) + lent
            if mode == 'same':
                pad1, pad2 = (pad + 2) // 2, (pad - 1) // 2
            else:
                pad1, pad2 = (pad + 1) // 2, pad // 2
            data = _pad_zeros(data, pad1, pad2)
            # in-place equivalent of
            # if demean:
            #     norm = ((_window_sum(data ** 2, lent) -
            #              _window_sum(data, lent) ** 2 / lent) * tnorm) ** 0.5
            # else:
            #      norm = (_window_sum(data ** 2, lent) * tnorm) ** 0.5
            # cc = cc / norm
            if demean:
                norm = _window_sum(data, lent) ** 2
                if norm.dtype == float:
                    norm /= lent
                else:
                    norm = norm / lent
                np.subtract(_window_sum(data ** 2, lent), norm, out=norm)
            else:
                norm = _window_sum(data ** 2, lent)
            norm *= tnorm
            if norm.dtype == float:
                np.sqrt(norm, out=norm)
                # print(norm)
            else:
                norm = np.sqrt(norm)
            mask = norm <= np.finfo(float).eps
            if cc.dtype == float:
                cc[~mask] /= norm[~mask]
            else:
                cc = cc / norm
            cc[mask] = 0
        else:
            msg = "normalize has to be one of (None, 'naive', 'full')"
            raise ValueError(msg)
    return cc

def cross0lag(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def vbardot(v):
    #z=[np.dot(v[x-l:x]-np.mean(v[x-l:x]),v[x-l:x]-np.mean(v[x-l:x])) for x in range(l,len(v))]
    return np.dot(v-np.mean(v),v-np.mean(v))

def cctop(template,snippet):
    out=np.dot(template,snippet) 
    return out

def generate_correlation_map(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])