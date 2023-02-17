#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:36:10 2021

@author: amande
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import re
import scipy
import glob

# load custom-written utility functions from file
from sibs_bburst_support_functions import *

######  GENERAL ANALYSIS SETTINGS
n_fft=1024
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
blo, bhi = 12, 35 # beta freq. limits 
lim = 100  # lower duration limit for beta burst, in ms
percentile=75
dwn=200 # downsampling target frequency (Hz)
fmin_plot=2
fmax_plot=48

# example subject selection and subject-specific information
subj=166
top_ch=7
top_freq=21
beta_lo=top_freq-1.5
beta_hi=top_freq+1.5
side='right'
# t_from=40 # start time (in seconds)
# t_to=45 # end time (in seconds)
t_from=20 # start time (in seconds)
t_to=25 # end time (in seconds)

######  PATH INFORMATION 
path_trunc='/main/'
path_to_data1=path_trunc + 'raw_data/standard/'
path_to_data2=path_trunc + 'raw_data/exceptions/'
path_to_figs=path_trunc + 'figures/'
path_metadata=path_trunc + 'metadata_tables/'

######  MEG DATA FILENAME INFORMATION & PATHS
txtfiles=[]
folders1=glob.glob(path_to_data1+'*')
for i in range(0,len(folders1)):
    tmp=glob.glob(folders1[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)    
folders1=glob.glob(path_to_data2+'*')
for i in range(0,len(folders1)):
    tmp=glob.glob(folders1[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)   
txtfiles=np.concatenate(txtfiles)

# define ROI
ROI=['MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123', 'MEG1122', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322',
            'MEG0723', 'MEG0722', 'MEG1142', 'MEG1143', 'MEG1133', 'MEG1132', 'MEG1342', 'MEG1343', 'MEG1333', 'MEG1332', 
            'MEG0733', 'MEG0732', 'MEG2212', 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422']
 
for h in range(0, len(txtfiles)):  
    ###### LOAD RAW DATA
    fn = txtfiles[h]
    fs = fn.split("/")
    tmp=re.split('(\d+)',fs[-2])
    pat_id=int(tmp[1])   
    times=fs[-3].split("_")
    if subj==pat_id:
        raw_eo = sib_load_data_1(fn, fs, times, txtfiles)
    
        chans=pick_pair(ROI, top_ch)      
      
        ######  BETA BURST ANALYSIS AND PLOTTING OF BASIC BURST METRICS         
        # high pass filter for MEG channels (gradiometers)
        picks_meg = mne.pick_types(raw_eo.info, meg='grad', eeg=False, stim=False, eog=False, emg=False,
                                exclude='bads')
        raw_eo.filter(fmin, None, picks=picks_meg, fir_design='firwin')
        
        # get maximum channels for futher analysis    
        raw, times = raw_eo.get_data(picks=chans, return_times=True)   
                        
        # get time-resolved power info using wavelets
        sfreq=raw_eo.info['sfreq']
    
        # obtain beta amplitude envelope
        def beta_amplitude_envelope(data, sfreq, dwn, beta_lo, beta_hi):
            # Resampling
            ### 16.9.2019 - Jussi Nurminen says mne.filter.resample includes filters to avoid aliasing so these can be used
            down=sfreq/dwn   
            # downsample to frequency specified by 'dwn'
            out1= mne.filter.resample(data, down=down, npad='auto', n_jobs=16, pad='reflect_limited', verbose=None) # additional options: window='boxcar', npad=100,
            
            # split data into consecutive epochs
            window=20 # length of individual time windows
            ws=int(window*dwn/fmin) # number of samples per window
            overlap=1-0 # set amount of overlap for consecutive FFT windows (second number sets amount of overlap)
            
            # separate data into consecutive data chunks (episode-like, because spectral_connectivity expects epochs)
            array1 = list()
            start = 0
            stop=ws
            step = int(ws*overlap)
            while stop < out1.shape[0]:
                tmp = out1[start:stop]
                start += step
                stop += step
                array1.append(tmp)
            array1=np.expand_dims(array1, axis=1)
            
            # define frequencies of interest
            freqs = np.arange(7., 47., 1.)
            n_cycles = freqs / 2.
            
            power = mne.time_frequency.tfr_array_morlet(array1, sfreq=dwn, freqs=freqs,
                                n_cycles=n_cycles, output='complex', n_jobs=16)
            b_idx=np.where(np.logical_and(freqs>beta_lo, freqs<beta_hi))
          
            amplitude=[]
            for k in range(0,len(power)):
                tmp=power[k][0][b_idx]
                tmptmp=np.mean(tmp, axis=0)
                amplitude=np.concatenate((amplitude,tmptmp), axis=None)
            
            return freqs, amplitude, out1  
    
        # get amplitude envelope 
        freqs, amp_ch1, out1 =beta_amplitude_envelope(raw[0], sfreq, dwn, beta_lo, beta_hi)
        freqs, amp_ch2, out2 =beta_amplitude_envelope(raw[1], sfreq, dwn, beta_lo, beta_hi)
        
        # at this stage, get vector sum time series
        def vector_sum(ts1, ts2):
            vs_ts=np.sqrt(np.square(ts1)+np.square(ts2))
            return vs_ts
        rec_amp_vectorsum=np.absolute(vector_sum(amp_ch1, amp_ch2))
        
        # smooth amplitude envelope and binarize data  
        fwhm=dwn/10 # downsampled frequency/10, i.e. 20 samples for 200 Hz (= 100 ms)
        
        def fwhm2sigma(fwhm):
            return fwhm / np.sqrt(8 * np.log(2))
        sigma = fwhm2sigma(fwhm)
    
        # filter    
        filt_vs=scipy.ndimage.filters.gaussian_filter1d(rec_amp_vectorsum, sigma, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
             
        # extract burst characteristics
        val=np.percentile(filt_vs, percentile)  
    
        # binarize
        bin_burst =(filt_vs > val).astype(np.int_) # outputs binarized data, 1 for values above threshold
    
        cutoff=np.ceil(lim/1000*dwn) ### --> multiply with sfreq to get value in data points
        burst_dur=[]
        burst_onset=[]
        burst_dur_ms=[]
        burst_amp=[]
        burst_offset=[]

        burst_info = rle(bin_burst) 
        for l in range(0,len(burst_info[0])):
            if burst_info[2][l]>0:
                if burst_info[0][l]>=cutoff:                            
                    tmp=burst_info[0][l]    # burst duration
                    tmp1=burst_info[1][l]   # burst onset
                    tmp2=tmp1+tmp           # burst offset                            
                    burst_dur=np.concatenate((burst_dur,tmp), axis=None)
                    burst_onset=np.concatenate((burst_onset,tmp1), axis=None)
                    burst_offset=np.concatenate((burst_offset, tmp2), axis=None)
        burst_dur_ms=(burst_dur/sfreq)*1000

        # binarized & temporally thresholded time series (bursts > lim)
        zeros = [0] * len(bin_burst)
        
        # binarized timeseries, all bursts
        burst_binary_ts=[]
        for k in range(0, len(burst_info[0])):
            tmp4=[]
            bdur=burst_info[0][k]
            bonset=burst_info[1][k]
            if burst_info[2][k]>0:
                if burst_info[0][k]>=cutoff:
                    tmp2=zeros[bonset:bonset+bdur] 
                    for x in tmp2:
                        tmp4.append(1)   # binarized & duration thresholded trace                            
                else:
                    tmp4=zeros[bonset:bonset+bdur] 
    
            else:
                tmp4=zeros[bonset:bonset+bdur]   
            burst_binary_ts=np.concatenate((burst_binary_ts,tmp4), axis=None)
             
        # Font specifications
        font = {'family': 'normal',
            'color':  'black',
            'weight': 'normal',
            'size': 24,
            }

        start=int(t_from*sfreq)
        stop=int(t_to*sfreq)
        time=np.arange(t_from, t_to, 1/sfreq)
        time=time[0:(stop-start)]
        
        ylim=np.max(filt_vs[start:stop])+(np.max(filt_vs[start:stop]))/20
        y1=burst_binary_ts[start:stop]                
                
        
        # Figure with raw data to amplitude envelope illustration
        fig, axs = plt.subplots(3, 1, figsize=(12, 6))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.02, right=0.95, top=0.92, wspace=0.25)

        # raw signal: out1
        ax1=plt.subplot(311)
        plt.plot(time, np.squeeze(out1)[start:stop], 'r', label='raw data')
        plt.plot(time, np.squeeze(out2)[start:stop]-1.5*np.max(np.squeeze(out1)[start:stop]), 'k', label='raw data')
        plt.title('raw data', loc='left', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        frame1.axes.get_xaxis().set_ticklabels([])
        
        # narrow band filtered signal: amplitude
        ax2=plt.subplot(312)
        plt.plot(time, amp_ch1[start:stop], 'r')
        plt.plot(time, amp_ch2[start:stop]-1.5*np.max(amp_ch1[start:stop]), 'k')
        plt.title('narrow band filtered data', loc='left', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        frame1.axes.get_xaxis().set_ticklabels([])
        
        # smoothed amplitude envelope: filt 1
        ax3=plt.subplot(313)
        plt.plot(time, filt_vs[start:stop], label='amplitude envelope')
        ax3.axhline(val, color='red', lw=2, alpha=0.5)
        ax3.fill_between(time, 0, ylim, y1 > 0,
                        facecolor='red', alpha=0.3)
        ax3.fill_between(time, val, filt_vs[start:stop], y1 > 0,
                        facecolor='red', alpha=0.4)
        plt.title('amplitude envelope', loc='left', fontdict=font)
        plt.xticks(fontsize=24)            
        plt.xlabel('time (s)', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        
        ftitle=(path_to_figs + 'Figure1B_raw_data_processing1_%s_%s.png' % (pat_id, side))          
        fig.savefig(ftitle) 


        # Figure with amplitude envelope example (for phenotype illustration)
        fig, axs = plt.subplots(1, 1, figsize=(12, 3))
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.3, left=0.02, right=0.95, top=0.95, wspace=0.25)
        
        ax3=plt.subplot(111)
        plt.plot(time, filt_vs[start:stop])
        ax3.axhline(val, color='red', lw=2, alpha=0.5)
        ax3.fill_between(time, 0, ylim, y1 > 0,
                        facecolor='red', alpha=0.3)
        ax3.fill_between(time, val, filt_vs[start:stop], y1 > 0,
                        facecolor='red', alpha=0.4)
        plt.xticks(fontsize=24)            
        plt.xlabel('time (s)', fontdict=font)
        frame1 = plt.gca()
        frame1.axes.get_yaxis().set_ticks([])
        plt.xticks(fontsize=24)            
        plt.xlabel('time (s)', fontdict=font)    
        
        ftitle=(path_to_figs + 'Figure1B_raw_data_processing2_%s_%s.png' % (pat_id, side))          
        fig.savefig(ftitle) 
        