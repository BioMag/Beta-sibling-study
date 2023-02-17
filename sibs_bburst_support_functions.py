#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:27:04 2021

@author: amande

This script is a repository of all the utility functions used for data analysis
in the other scripts. 

Basic routines include loading of data, combining two PSD spectra into one vector
PSD, removing aperiodic signal component using FOOOF, obtaining the beta range
amplitude envelope, extracting beta event characteristics from the beta envelope as
well as some plotting functions. 

"""
import mne
import re             
import numpy as np
from fooof import FOOOF
import scipy
n_exceptions=12
               
# function to load two conditions (the MOT and EO conditions) 
def sib_load_data(fn, fs, times, txtfiles, idx_files):
    if  fs[4]=='standard':
        # data path contains timing information: 
        # order of conditions EC, EO, hand movement unless otherwise stated
        # first pair of numbers EC, second EO, third hand

        if len(times)<9:
            # load eo data
            tmin=float(times[2])
            tmax=float(times[3])
            tmin1=float(times[4])
            tmax1=float(times[5])
            
            # load EO data
            raw_eo=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
             
            # load MOT data
            raw_mot=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()
            
        else:
            raws=list()
            
            # load EO data
            tmin=float(times[2])
            tmax=float(times[9])
            tmin1=float(times[10])
            tmax1=float(times[3])    
            
            raw1=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
            raw2=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()
            raws.append(raw1)
            raws.append(raw2)
            raw_eo=mne.concatenate_raws(raws)
                                
            # load MOT data
            tmin1=float(times[4])
            tmax1=float(times[5])
            raw_mot=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()

    if  fs[4]=='exceptions':
            # exceptions are files where the data was collected in seperate files. In this case,
            # the maching second file needs to be found to get a pair EO + HAND
            # there are n_exceptions such data sets (one containing EO+ EC, the other the HAND condition)
            
            tmp=re.split('(\d+)',fs[-2])
            pat_id=int(tmp[1])   
            
            tmin1=float(times[2])
            tmax1=float(times[3])
             
            # load MOT data
            raw_eo=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()
            #raw_mot=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()   
            
            # matching EO data file
            fn2=txtfiles[idx_files+n_exceptions]
            fs2 = fn2.split("/")
            tmp=re.split('(\d+)',fs2[-2])
            pat_id2=int(tmp[1])   
            
            if pat_id != pat_id2:
                raise RuntimeError('subject id not matching, script aborted')
                
            times2=fs2[-3].split("_")
            tmin=float(times2[4])
            tmax=float(times2[5])
            
            # load EO data
            #raw_eo=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
            raw_mot=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data() 
            
    return raw_mot, raw_eo


# function to load only one condition (the EO condition)
def sib_load_data_1(fn, fs, times, txtfiles):
    if  fs[4]=='standard':
        # data path contains timing information: 
        # order of conditions EC, EO, hand movement unless otherwise stated
        # first pair of numbers EC, second EO, third hand

        if len(times)<9:
            # load eo data
            tmin=float(times[2])
            tmax=float(times[3])
            
            # load EO data
            raw_eo=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
             
        else:
            raws=list()
            
            # load EO data
            tmin=float(times[2])
            tmax=float(times[9])
            tmin1=float(times[10])
            tmax1=float(times[3])    
            
            raw1=mne.io.read_raw_fif(fn).crop(tmin, tmax).load_data()
            raw2=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()
            raws.append(raw1)
            raws.append(raw2)
            raw_eo=mne.concatenate_raws(raws)
                                
    if  fs[4]=='exceptions':
            # exceptions are files where the data was collected in seperate files. In this case,
            # the maching second file needs to be found to get a pair EO + HAND
            # there are n_exceptions such data sets (one containing EO+ EC, the other the HAND condition)
            tmin1=float(times[2])
            tmax1=float(times[3])
             
            # load EO data
            raw_eo=mne.io.read_raw_fif(fn).crop(tmin1, tmax1).load_data()

    return raw_eo

# vector sum calculation for a set of PSDs in the ROI  
def vector_sum(power, roi):
    pairs=np.arange(0,len(roi),2)     
    vs=list()
    labels=list()
    for i in range(0, len(pairs)):
        idx=pairs[i]
        tmp=np.sqrt(np.square(power[idx])+np.square(power[idx+1]))
        tmp2=[roi[idx],roi[idx+1]]
        vs.append(tmp)
        labels.append(tmp2)
    return vs, labels

# vector sum calculation for two time series
def vector_sum_ts(ts1, ts2):
    vs_ts=np.sqrt(np.square(ts1)+np.square(ts2))
    return vs_ts

# Use FOOOF to de-trend data (an obtain only periodic component of spectrum)
####  FOOOF background on aperiodic component ('1/f noise')
# 𝑏, 𝑘, and 𝜒 of the aperiodic component which reflect the offset, knee and exponent, respectively
# if using linear space: 10^b ∗ 1/(𝑘+𝐹𝜒)
def remove_aperiodic(freqs, data, freq_range):
    PERIODIC=list()
    APERIODIC=list()
    APERIODIC_b=list()
    APERIODIC_x=list()
    for i in range(0,len(data)):                        
        # Initialize a FOOOF object
        fm = FOOOF()
        
        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        fm.fit(freqs, data[i], freq_range)
        print('Aperiodic parameters: \n', fm.aperiodic_params_, '\n')
        print('Peak parameters: \n', fm.peak_params_, '\n')
        
        # test 1/f noise
        b=fm.aperiodic_params_[0]
        x=fm.aperiodic_params_[1]
        
        aperiodic=(np.power(10, b)) * (1/np.power(freqs, x))
        periodic = np.subtract(data[i], aperiodic) 
        PERIODIC.append(periodic)
        APERIODIC.append(aperiodic)
        APERIODIC_b.append(b)
        APERIODIC_x.append( x)
    return PERIODIC, APERIODIC, APERIODIC_b, APERIODIC_x

# routine to get maximum beta channel/frequenc/amplitude
def find_bmax_info(periodic, aperiodic_x, aperiodic_b, f, beta_lo, beta_hi, force_ch, idx):
    # indices of data values within individual beta peak frequency
    # the individual maximum will be determined from these
    beta_idx=np.where(np.logical_and(f>beta_lo, f<beta_hi))
    Max_amp=list()
    Max_freq=list()
    for amp in range(0, len(periodic)):
        # print(amp)
        # find amplitude maximum in individual beta frequency range per channel
        if np.any(np.isnan(periodic[amp][beta_idx]))== True:
            max_amp=[]   
            max_freq=[]     
        else:
            # get frequency at which maximum amplitude occurs
            max_amp=np.max(periodic[amp][beta_idx])
            idx_max_freq=np.argwhere(periodic[amp][beta_idx]==max_amp)
            tmp=beta_idx[0][int(idx_max_freq)] # beta_idx is a tuple which is why the [0] index is necessary
            max_freq=f[tmp]                      
        Max_amp.append(max_amp) # maximum beta ampltude (for all channels, in the manually chosen frequency range)
        Max_freq.append(max_freq) # maximum beta frequency (for all channels, in the manually chosen frequency range)
    
    # get maximum amplitude, maximum frequency and maximum channel
    beta_max_amp=np.max(Max_amp) # maximum beta amplitude across all channels
    beta_max_ch=int(np.argwhere(Max_amp==beta_max_amp)) # channel which has maximum beta amplitude
    beta_max_freq=Max_freq[beta_max_ch] # frequency which has maximum beta amplitude in this channel
    noise_comp=aperiodic_x[beta_max_ch] # aperiodic chi of 1/f component for this channel
    offset=aperiodic_b[beta_max_ch] # aperiodic offset of 1/f component for this channel
    if idx==0 and np.isnan(force_ch)!= True:
        beta_max_ch=int(force_ch)
        beta_max_freq=Max_freq[beta_max_ch]
        beta_max_amp=Max_amp[beta_max_ch]
        noise_comp=aperiodic_x[beta_max_ch] # aperiodic chi of 1/f component for this channel
        offset=aperiodic_b[beta_max_ch] # aperiodic offset of 1/f component for this channel
            
    return(beta_max_freq, beta_max_amp, beta_max_ch, noise_comp, offset)

# routine to get beta frequency range power
def get_total_beta_power(vs, periodic, beta_range, beta_max_ch, f):
    beta_idx=np.where(np.logical_and(f>beta_range[0], f<beta_range[1]))
    tot_beta=np.sum(vs[beta_max_ch][beta_idx])
    tot_periodic_beta=np.sum(periodic[beta_max_ch][beta_idx])
    return tot_beta, tot_periodic_beta

# determine burst duration and burst onset times
### copied from stackoverflow
### https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
    ## run length encoding. Partial credit to R rle function. 
    ## Multi datatype arrays catered for including non Numpy
    ## returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i]) # return(z, p, ia[i])

# routine to pick a channel pair from ROI list using the index from file
# the index is indexing into the vector PSD with the highest beta range peak
# selected previously
def pick_pair(ROI, top_ch):
    ch1= ROI[top_ch*2]
    ch2= ROI[top_ch*2 + 1]
    chans=[ch1, ch2]
    return chans

# get time-resolved power info using wavelets
# obtain beta amplitude envelope
def beta_amplitude_envelope(data, sfreq, dwn, top_freq, fmin):
    # Resampling
    ### check with Jan Kujala/Jussi Nurminen what this does to the data (and whether we want to do it)
    ### 16.9.2019 - Jussi Nurminen says mne.filter.resample includes filters to avoid aliasing 
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

    # calculate power        
    power = mne.time_frequency.tfr_array_morlet(array1, sfreq=dwn, freqs=freqs,
                        n_cycles=n_cycles, output='complex', n_jobs=16)
    
    beta_lo=top_freq-1.5
    beta_hi=top_freq+1.5
    b_idx=np.where(np.logical_and(freqs>beta_lo, freqs<beta_hi))
  
    amplitude=[]
    for k in range(0,len(power)):
        tmp=power[k][0][b_idx]
        tmptmp=np.mean(tmp, axis=0)
        amplitude=np.concatenate((amplitude,tmptmp), axis=None)
    
    return freqs, amplitude   

# extract burst characteristics
def get_burst_characteristics(filt, percentile, lim, dwn):
    val=np.percentile(filt, percentile)  

    # binarize
    bin_burst =(filt > val).astype(np.int_) # outputs binarized data, 1 for values above threshold
    
    burst_dur=[]
    burst_dur_ms=[]
    burst_amp=[]
    burst_onset=[]
    burst_offset=[]
    cutoff=np.ceil(lim/1000*dwn) ### --> multiply with sfreq to get value in data points

    # basic burst characteristics (burst duration & amplitude)
    burst_info = rle(bin_burst) 
    for l in range(0,len(burst_info[0])):
        if burst_info[2][l]>0:
            if burst_info[0][l]>=cutoff:
                tmp=burst_info[0][l]    # burst duration
                tmp1=burst_info[1][l]   # burst onset
                tmp2=tmp1+tmp           # burst offset
                tmp3=np.max(filt[tmp1:tmp1+tmp]) # burst amplitude
                burst_dur=np.concatenate((burst_dur,tmp), axis=None)
                burst_onset=np.concatenate((burst_onset,tmp1), axis=None)
                burst_amp=np.concatenate((burst_amp,tmp3), axis=None)
                burst_offset=np.concatenate((burst_offset, tmp2), axis=None)
    burst_dur_ms=(burst_dur/dwn)*1000
    
    ibi=[]
    for l in range(1,len(burst_offset)):
        tmp=burst_onset[l]-burst_offset[l-1]
        ibi=np.concatenate((ibi,tmp), axis=None)
    ibi=(ibi/dwn)*1000   # inter-burst interval (in ms)               
    di=np.sqrt(np.var(ibi))/np.mean(ibi) # dispersion index
    bps=len(burst_dur_ms)/(len(filt)/dwn)  # bursts per second
    
    return burst_info, burst_amp, burst_dur_ms, ibi, di, bps

# routine to extract basic burst descriptives
def get_descriptives(data, max_percentile):
    dmean=np.mean(data)
    dmedian=np.median(data)
    dstdev=np.std(data)
    dmax=np.max(data)
    max_lim=np.percentile(data, max_percentile)  
    max_idx=np.asarray(np.where(data>max_lim))
    drobustmax=np.mean(data[max_idx])
    return (dmean, dmedian, dstdev, dmax, drobustmax)

# extract burst information from vector amplitude envelope
def get_burst_info(top_ch, top_freq, ROI, raw, percentile, lim, dwn, fmin):

    chans=pick_pair(ROI, top_ch)    
      
    ######  BETA BURST ANALYSIS AND PLOTTING OF BASIC BURST METRICS 
    sfreq=raw.info['sfreq']
    # notch filtering for all channels
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, emg=True,
                             exclude='bads')
    raw.notch_filter(np.arange(50, 240, 50), picks=picks, fir_design='firwin')
    
    # low pass filter for MEG channels (gradiometers)
    picks_meg = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=False, emg=False,
                            exclude='bads')
    raw.filter(fmin, None, picks=picks_meg, fir_design='firwin')
    
    # get maximum channels for futher analysis    
    raw, times = raw.get_data(picks=chans, return_times=True)   
                    
    # get time-resolved power info (amplitude envelope)
    freqs, amp_ch1=beta_amplitude_envelope(raw[0], sfreq, dwn, top_freq, fmin)
    freqs, amp_ch2=beta_amplitude_envelope(raw[1], sfreq, dwn, top_freq, fmin)
    
    # get vector sum time series (i.e. both amplitude envelope timeseries are 
    # combined into one from here on)
    # this cannot be done at any earlier stage because the vector operations 
    # rectify the data which fundamentally alters the raw data structure
    # the amplitude envelope is rectified anyway so this makes sense at this stage
    rec_amp_vectorsum=np.absolute(vector_sum_ts(amp_ch1, amp_ch2))
    
    # smooth amplitude envelope and binarize data  
    fwhm=dwn/10 # downsampled frequency/10, i.e. 20 samples for 200 Hz (= 100 ms)
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))
    sigma = fwhm2sigma(fwhm)    
    filt_vs=scipy.ndimage.filters.gaussian_filter1d(rec_amp_vectorsum, sigma, axis=-1, order=0, mode='reflect', cval=0.0, truncate=4.0)
        
    # binarize and threshold amplitude envelope, extract burst characteristics            
    burst_info, burst_amp, burst_dur_ms, ibi, di, bps=get_burst_characteristics(filt_vs, percentile, lim, dwn)    

    # extract basic descriptives
    max_percentile=95 # percentile cutoff for robust mean estimation
    dmean1, dmedian1, dstdev1, dmax1, drobustmax1=get_descriptives(burst_amp, max_percentile)
    dmean2, dmedian2, dstdev2, dmax2, drobustmax2=get_descriptives(burst_dur_ms, max_percentile)
    
    return dmean1, dmedian1, dstdev1, dmax1, drobustmax1, dmean2, dmedian2, dstdev2, dmax2, drobustmax2, bps, di
