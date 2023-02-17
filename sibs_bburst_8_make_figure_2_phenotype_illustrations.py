#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:31:34 2020

@author: paulsk
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import mne
import re

# load custom-written utility functions from file
from sibs_bburst_support_functions import *

######  GENERAL ANALYSIS SETTINGS
n_fft=1024
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
freq_range=[fmin, fmax]

# font settings for plotting
fontsize=28
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : fontsize}

# plotting settings
fmin_plot=13
fmax_plot=28
 
######  PATH INFORMATION 
path_trunc='/main/'
path_to_data1=path_trunc + 'raw_data/standard/'
path_to_data2=path_trunc + 'raw_data/exceptions/'
path_to_figs=path_trunc + 'figures/'
path_metadata=path_trunc + 'metadata_tables/'

######  SUBJECT METADATA
df_meta = pd.read_csv(path_metadata + "gender_age_sibling_ap.csv") 

######  MEG DATA FILENAME INFORMATION & PATHS
import glob

txtfiles=[]
folders1=glob.glob(path_to_data1+'*')
for i in range(0,len(folders1)):
    tmp=glob.glob(folders1[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)    
folders2=glob.glob(path_to_data2+'*')
for i in range(0,len(folders2)):
    tmp=glob.glob(folders2[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)   
txtfiles=np.concatenate(txtfiles)
n_exceptions=12 # number of files where EO and MOT data are recorded in separate files

# define ROI
roi_left=['MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0633', 'MEG0632',
          'MEG0243', 'MEG0242', 'MEG0232', 'MEG0233', 'MEG0443', 'MEG0442', 'MEG0432', 'MEG0433', 'MEG0713', 'MEG0712', 
          'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1813', 'MEG1812', 'MEG1822', 'MEG1823', 'MEG0743', 'MEG0742',]

roi_right=['MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123', 'MEG1122', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322',
           'MEG0723', 'MEG0722', 'MEG1142', 'MEG1143', 'MEG1133', 'MEG1132', 'MEG1342', 'MEG1343', 'MEG1333', 'MEG1332', 
           'MEG0733', 'MEG0732', 'MEG2212', 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422']


# beta phenotype illustrations
sib_pairs=[6, 46, 50]
subject_numbers=[1, 1, 1]
sides=['right', 'right', 'left']
channels=[12, 13, 11]

periodic_all=list()
frequencies=list()

for p in range(0, len(sib_pairs)):
    sib_pair=sib_pairs[p]
    side=sides[p]
    channel=channels[p]
    subject_number=subject_numbers[p]
    pair_id=df_meta.loc[df_meta['Pair'] == sib_pair].to_numpy()
    subj=pair_id[subject_number][0]
    sib=pair_id[subject_number][2]
    for h in range(0, len(txtfiles)-n_exceptions):  
        ###### LOAD RAW DATA
        fn = txtfiles[h]
        fs = fn.split("/")
        tmp=re.split('(\d+)',fs[-2])
        pat_id=int(tmp[1])   
        if subj==pat_id:
            times=fs[-3].split("_")
            
            # load raw data
            raw_eo = sib_load_data_1(fn, fs, times, txtfiles)
                
            sfreq=raw_eo.info['sfreq']
            if side=='left':
                picks=mne.pick_channels(raw_eo.info['ch_names'], include=roi_left,
                        exclude=[], ordered=True)
                roi=roi_left
            else:
                picks=mne.pick_channels(raw_eo.info['ch_names'], include=roi_right,
                        exclude=[], ordered=True)
                roi=roi_right

            # filtering (fmin, fmax defined earlier)
            raw_eo.filter(fmin, fmax, fir_design='firwin')        
            
            # calaculate power spectra for ROI
            [power1,f] = mne.time_frequency.psd_welch(raw_eo, picks=picks, tmin=None, tmax=None,
                                    fmin=fmin, fmax=fmax, n_fft=n_fft)
                
            # vector sum calculation       
            pairs=np.arange(0,len(roi),2)     
            vs=list()

            for i in range(0, len(pairs)):
                idx=pairs[i]
                tmp=np.sqrt(np.square(power1[idx])+np.square(power1[idx+1]))
                tmp2=roi_left[idx]
                tmp3=roi_left[idx+1]
                tmp4=[tmp2,tmp3]
                vs.append(tmp)
                
            ######## SEPARATE APERIODIC & PERIODIC COMPONENT ###############        
            ####  FOOOF background on aperiodic component ('1/f component')
            # 𝑏, 𝑘, and 𝜒 of the aperiodic component which reflect the offset, knee and exponent, respectively
            # if using linear space: 10^b ∗ 1/(𝑘+𝐹𝜒)
        
            periodic, aperiodic, aperiodic_b, aperiodic_x = remove_aperiodic(f, vs, freq_range)
            periodic_all.append(periodic[channel])
            frequencies=f
            
            
########## PLOT PERIODIC COMPONENT BETA EXAMPLES ################        
b_idx=np.squeeze(np.where(np.logical_and(frequencies>fmin_plot,frequencies<fmax_plot)))
        
# plot all vector sum PSD spectrum periodic component   
fig, axs = plt.subplots(1, len(periodic_all), figsize=(15, 5), facecolor='w', edgecolor='k')
for i in range(0, len(periodic_all)):
    axs[i].plot(f[b_idx],periodic_all[i][b_idx], 'k', linewidth=4)
    axs[i].yaxis.set_ticklabels([]) 
    axs[i].tick_params(labelsize=fontsize)
fig.tight_layout()
ftitle=('Figure2_beta_phenotype_examples_raw_data.png')
fname  = path_to_figs + ftitle
plt.gcf().savefig(fname)


# plots illustration similarity and segregation           
sib_pairs=[5, 6, 90, 72, 85, 98, 35, 99]
sides=['left', 'right', 'left', 'right', 'left', 'right', 'right', 'right']
channels=[11, 12, 12, 12, 13, 7, 13, 12]
examples_seg_sim_all=list()
for p in range(0, len(sib_pairs)):
    sib_pair=sib_pairs[p]
    side=sides[p]
    channel=channels[p]
    pair_id=df_meta.loc[df_meta['Pair'] == sib_pair].to_numpy()
    RAW=list()
    for g in range(0, len(pair_id)):
        subj=pair_id[g][0]
        for h in range(0, len(txtfiles)-n_exceptions):
            ###### LOAD RAW DATA
            fn = txtfiles[h]
            fs = fn.split("/")
            tmp=re.split('(\d+)',fs[-2])
            pat_id=int(tmp[1])   
            if subj==pat_id:
                times=fs[-3].split("_")
            
                # load raw data
                raw_eo = sib_load_data_1(fn, fs, times, txtfiles)
                RAW.append(raw_eo)             
                        
    ########################### POWER CALCULATION, VECTOR SUM CALCULATION ##########################################
    periodic_all=list()
    for g in range(0, len(RAW)):
        raw_eo=RAW[g]
        sfreq=raw_eo.info['sfreq']
        if side=='left':
            # define subset of channels to use
            picks=mne.pick_channels(raw_eo.info['ch_names'], include=roi_left,
                exclude=[], ordered=True)
            roi=roi_left
        else:
            picks=mne.pick_channels(raw_eo.info['ch_names'], include=roi_right,
                exclude=[], ordered=True)
            roi=roi_right
    
        # filtering (fmin, fmax defined earlier)
        raw_eo.filter(fmin, fmax, fir_design='firwin')
        
        # calaculate power spectra for ROI
        [power1,f] = mne.time_frequency.psd_welch(raw_eo, picks=picks, tmin=None, tmax=None,
                                fmin=fmin, fmax=fmax, n_fft=n_fft)
            
        # vector sum calculation       
        pairs=np.arange(0,len(roi),2)     
        vs=list()
        for i in range(0, len(pairs)):
            idx=pairs[i]
            tmp=np.sqrt(np.square(power1[idx])+np.square(power1[idx+1]))
            tmp2=roi[idx]
            tmp3=roi[idx+1]
            tmp4=[tmp2,tmp3]
            vs.append(tmp)
            
        ######## SEPARATE APERIODIC & PERIODIC COMPONENT ###############
        periodic, aperiodic, aperiodic_b, aperiodic_x = remove_aperiodic(f, vs, freq_range)
        periodic_all.append(periodic[channel])
        frequencies=f                   

    examples_seg_sim_all.append(periodic_all)
    
    
######### PLOT BETA EXAMPLES FOR SIBLING PAIRS ##################
# example figure combining all of the chosen example plots
fig, axs = plt.subplots(2, int(np.round(len(examples_seg_sim_all)/2)), figsize=(20, 10), facecolor='w', edgecolor='k')
axs = axs.ravel()
for i in range(len(examples_seg_sim_all)):
    if len(examples_seg_sim_all[i])==2:
        smoothed1 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][0])
        smoothed2 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][1])
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][0][b_idx], 'k', linewidth=4)
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][1][b_idx], 'r', linewidth=4)
        axs[i].yaxis.set_ticklabels([]) 
        axs[i].tick_params(labelsize=fontsize)
    if len(examples_seg_sim_all[i])==3:
        smoothed1 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][0])
        smoothed2 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][1])
        smoothed3 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][2])
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][0][b_idx], 'k', linewidth=4)
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][1][b_idx], 'r', linewidth=4)
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][2][b_idx], 'b', linewidth=4)
        axs[i].yaxis.set_ticklabels([]) 
        axs[i].tick_params(labelsize=fontsize)
    if len(examples_seg_sim_all[i])==4:
        smoothed1 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][0])
        smoothed2 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][1])
        smoothed3 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][2])
        smoothed4 = scipy.signal.filtfilt([1,1],2, examples_seg_sim_all[i][3])
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][0][b_idx], 'k', linewidth=4, label='sib 1')
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][1][b_idx], 'r', linewidth=4, label='sib 2')
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][2][b_idx], 'b', linewidth=4, label='sib 3')
        axs[i].plot(frequencies[b_idx],examples_seg_sim_all[i][3][b_idx], 'g', linewidth=4, label='sib 4')
        axs[i].legend(fontsize=fontsize)
        axs[i].yaxis.set_ticklabels([]) 
        axs[i].tick_params(labelsize=fontsize)
        
fig.tight_layout()
ftitle=('Figure2_beta_similarity_and_segregation_examples_raw_only.png')
fname  = path_to_figs + ftitle
plt.gcf().savefig(fname)

