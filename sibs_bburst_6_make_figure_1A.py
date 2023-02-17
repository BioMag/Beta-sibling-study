#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:36:10 2021

@author: amande
"""

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import mne
import os
import re

# load custom-written utility functions from file
from sibs_bburst_support_functions import *

######  GENERAL ANALYSIS SETTINGS
n_fft=1024
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
blo, bhi = 12, 35 # beta freq. limits 
lim = 50  # lower duration limit for beta burst, in ms
percentile=[75]
dwn=200 # downsampling target frequency (Hz)
fmin_plot=2
fmax_plot=48

######  PATH INFORMATION
path_trunc='/main/'
path_to_data1=path_trunc + 'raw_data/standard/'
path_to_data2=path_trunc + 'raw_data/exceptions/'
path_to_figs=path_trunc + 'figures/'
path_metadata=path_trunc + 'metadata_tables/'

######  MEG DATA FILENAME INFORMATION & PATHS
import glob

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
# ROI=['MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0633', 'MEG0632',
#               'MEG0243', 'MEG0242', 'MEG0232', 'MEG0233', 'MEG0443', 'MEG0442', 'MEG0432', 'MEG0433', 'MEG0713', 'MEG0712', 
#               'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1813', 'MEG1812', 'MEG1822', 'MEG1823', 'MEG0743', 'MEG0742',]

subj=166
top_ch=7
top_freq=21

for h in range(0, len(txtfiles)):  
    ###### LOAD RAW DATA
    fn = txtfiles[h]
    fs = fn.split("/")
    tmp=re.split('(\d+)',fs[-2])
    pat_id=int(tmp[1])   
    if subj==pat_id:
        times=fs[-3].split("_")
        raw_mot, raw_eo = sib_load_data(fn, fs, times, txtfiles, h)

        #################### COMPUTE POWER SPECTRAL DENSITY (PSD) & TOPOPLOTS ##########################################               
        # ###### EO DATA #########
        modality='topoplots'
        sfreq=raw_eo.info['sfreq']
            
        # pick subset of channels (gradiometers)
        picks_meg = mne.pick_types(raw_eo.info, meg='grad', eeg=False, stim=False, eog=False, emg=False, exclude='bads')
        ch_names = np.asarray(raw_eo.ch_names)
        channels=ch_names[picks_meg]
        raw_chans=raw_eo.pick_channels(channels)
        
        # filtering (fmin, fmax defined earlier)
        raw_chans.filter(fmin, fmax, fir_design='firwin')
                
        # calculate PSD
        psds, freqs = mne.time_frequency.psd_welch(raw_chans, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)
        
        # ###### MOT DATA #########
        sfreq=raw_mot.info['sfreq']
            
        # pick subset of channels (gradiometers)
        picks_meg = mne.pick_types(raw_mot.info, meg='grad', eeg=False, stim=False, eog=False, emg=False, exclude='bads')
        ch_names = np.asarray(raw_mot.ch_names)
        channels=ch_names[picks_meg]
        raw_chans=raw_mot.pick_channels(channels)
        
        # filtering (fmin, fmax defined earlier)
        raw_chans.filter(fmin, fmax, fir_design='firwin')
        
        # calculate PSD
        psds2, freqs = mne.time_frequency.psd_welch(raw_chans, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)

        ######################################### PSD TOPOPLOT #########################################################        
        b_idx=np.where(np.logical_and(freqs>fmin_plot, freqs<fmax_plot))
        
        def my_callback(ax, ch_idx):
            """
            This block of code is executed once you click on one of the channel axes
            in the plot. To work with the viz internals, this function should only take
            two parameters, the axis and the channel or data index.
            """
            ax.plot(freqs, psds[ch_idx], color='red')
            ax.set_xlabel = 'Frequency (Hz)'
            ax.set_ylabel = 'Power (dB)'
        
        for ax, idx in mne.viz.iter_topography(raw_chans.info,
                                        fig_facecolor='white',
                                        axis_facecolor='white',
                                        axis_spinecolor='white',
                                        on_pick=my_callback):
            ax.plot(psds[idx][b_idx], color='red', label='REST')
            ax.plot(psds2[idx][b_idx], color='green', label='MOT')
    
        plt.gcf().set_size_inches(10, 10) 
        #plt.legend(fontsize=50, loc='best')

        ftitle=('/Figure1A_topo_subject_%s_%s-%sHz.pdf' % (subj, fmin_plot, fmax_plot)) 
        fname  = path_to_figs + ftitle
        plt.gcf().savefig(fname)
        plt.show() 
                 
        ########################### POWER CALCULATION, VECTOR SUM CALCULATION ##########################################               
        sfreq=raw_eo.info['sfreq']
    
        # define subset of channels to use
        picks_right=mne.pick_channels(raw_eo.info['ch_names'], include=ROI,
                exclude=[], ordered=True)

        # filtering (fmin, fmax defined earlier)
        raw_eo.filter(fmin, fmax, fir_design='firwin')
        raw_mot.filter(fmin, fmax, fir_design='firwin')
        
        # calaculate power spectra for ROI
        [power3,f] = mne.time_frequency.psd_welch(raw_eo, picks=picks_right, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)

        [power4,f] = mne.time_frequency.psd_welch(raw_mot, picks=picks_right, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)
        
        # vector sum calculation                   
        pairs=np.arange(0,len(ROI),2)     
        vs_right=list()
        vs_right_mot=list()
        labels_right=list()
        for i in range(0, len(pairs)):
            idx=pairs[i]
            tmp=np.sqrt(np.square(power3[idx])+np.square(power3[idx+1]))
            tmp1=np.sqrt(np.square(power4[idx])+np.square(power4[idx+1]))
            tmp2=ROI[idx]
            tmp3=ROI[idx+1]
            tmp4=[tmp2,tmp3]
            vs_right.append(tmp)
            vs_right_mot.append(tmp1)
            labels_right.append(tmp4)

        ######## SEPARATE NOISE & PERIODIC COMPONENT ###############
        PERIODIC_r, APERIODIC_r, APERIODIC_b_r, APERIODIC_x_r  = remove_aperiodic(f, vs_right, [2, 48])
        PERIODIC_r_mot, APERIODIC_r_mot, APERIODIC_b_r_mot, APERIODIC_x_r_mot = remove_aperiodic(f, vs_right_mot, [2, 48])
        
        ########## PLOT VECTOR SUM SPECTRA, EO & MOT, SIDE BY SIDE ######
        b_idx=np.where(np.logical_and(f>fmin_plot, f<fmax_plot)) 
        
        # ONE CHANNEL, RIGHT
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 24}
        beta_lim_low=14
        beta_lim_high=30
        tot_power_idx=np.where(np.logical_and(f>beta_lim_low, f<beta_lim_high)) 

        fig, axs = plt.subplots(1,1, figsize=(10, 7), facecolor='w', edgecolor='k')
        axs.plot(f[b_idx],vs_right[top_ch][b_idx], 'r', linewidth=5, label='raw PSD')
        axs.plot(f[b_idx],APERIODIC_r[top_ch][b_idx], 'k', linewidth=5, label='1/f')      
        axs.fill_between(f[tot_power_idx], APERIODIC_r[top_ch][tot_power_idx], vs_right[top_ch][tot_power_idx],
        facecolor='black', alpha=0.3)
        axs.legend(fontsize=24)
        axs.set_ylim(bottom=0, top=None)
        plt.xlabel('frequency (Hz)', font=font)
        plt.ylabel('PSD (au)', font=font)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        ftitle=('Figure1A_FOOOF_single_chan_right_fft%s_%s.png' % (n_fft, subj)) 
        fname  = path_to_figs + ftitle
        plt.gcf().savefig(fname, bbox_inches='tight')     
        
        
        # ROI FOOOF PLOT, RIGHT
        fig, axs = plt.subplots(3,5, figsize=(15, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)
        yr=list()
        for i in range(0, len(PERIODIC_r)):
            tmp=PERIODIC_r[i][b_idx]
            yr.append(tmp)
        ymax=np.max(yr)+np.max(yr)/10
        
        axs = axs.ravel()
        for i in range(len(PERIODIC_r)):
            axs[i].plot(f[b_idx],PERIODIC_r[i][b_idx], 'r', label='REST', linewidth=3)
            axs[i].plot(f[b_idx],PERIODIC_r_mot[i][b_idx], 'g', label='MOT', linewidth=3) 
            axs[i].yaxis.set_ticklabels([])                   
            axs[i].set_ylim(0,ymax)
        ftitle=('/Figure1A_perdiodic_indiv_right_fft%s_%s.png' % (n_fft, subj)) 
        fname  = path_to_figs + ftitle
        plt.gcf().savefig(fname)    
    