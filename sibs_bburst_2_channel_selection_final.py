#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:27:04 2021

Using the manually pre-selected beta frequency range peaks, the script goes through all 
files individually and finds the peak betachannel, peak beta frequency and amplitude.
It also saves information about the aperiodic compent (i.e. the exponential decay information)
from the power spectrum.

INPUT :
-------
1. Raw sibling MEG data found under
path_to_data1='/Volumes/LaCie/sibling_data/standard/'
path_to_data2='/Volumes/LaCie/sibling_data/exceptions/'

2. Meta data (df_meta) in a data frame which is saved under "sibs_beta_phenotypes_manual.csv"
This contains, per subject
['Number', - subject number
 'Age', - subject age
 'Sibling', - list of siblings (their id in Number)
 'Gender', - gender info
 'Pair', - subject pair number
 'Number.1', - subject number again (there to facilitate initial manual data entry)
 'ch_left', - left peak channel (this is revised in the script)
 'freq_left' - manually estimated peak frequency, left
 'Number.2', - subject number again (there to facilitate initial manual data entry)
 'ch_right', - right peak channel (this is revised in the script)
 'freq_right', - manually estimated peak frequency, right
 'force_ch_left', - in some cases, the channel is determined manually ('forced')
 'force_ch_right'] - in some cases, the channel is determined manually ('forced')

OUTPUT :
-------
Pandas data frame named 'sibling_bpower.csv', wich contains for both hemispheres 
(shown here for the left):
                    'Pair number': Sibling_pair,
                    'Subject number': Subject,
                    'MEG data subject number': MEG_data_id,
                    'Siblings': Sibling,
                    'Age': Age,
                    'Sex': Gender,
                    'peak channel left': beta_peak_ch_l,
                    'peak freq left': beta_peak_freq_l,
                    'peak amp left': beta_peak_amp_l,
                    'aperiodic left': noise_l,
                    'offset left': offset_l
                    'peak channel right': beta_peak_ch_r,
                    'peak freq right': beta_peak_freq_r,
                    'peak amp right': beta_peak_amp_r,   
                    'aperiodic right': noise_r
    
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import re
from fooof import FOOOF

# load custom-written utility functions from file
from sibs_bburst_support_functions import *

######  GENERAL ANALYSIS SETTINGS
n_fft=1024
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
freq_range = [fmin, fmax]
beta_range = [14, 30]
lim = 50  # lower duration limit for beta burst, in ms
percentile=75

# set plotting range
fmin_plot=14
fmax_plot=30

######  PATH INFORMATION 
path_trunc='/main/'
path_to_data1=path_trunc + 'raw_data/standard/'
path_to_data2=path_trunc + 'raw_data/exceptions/'
path_to_figs=path_trunc + 'figures/'
path_metadata=path_trunc + 'metadata_tables/'

###### FILE NAMES FOR SAVING
fn_dataframe='sibling_beta_PSD_phenotype.csv'

######  LOAD SUBJECT METADATA
df_meta = pd.read_csv(path_metadata + "sibling_beta_phenotypes_manual.csv") 
df_meta = df_meta.drop([213]) # removes the last line which contains NaNs (i.e. is empty)

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
roi_left=['MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0633', 'MEG0632',
          'MEG0243', 'MEG0242', 'MEG0232', 'MEG0233', 'MEG0443', 'MEG0442', 'MEG0432', 'MEG0433', 'MEG0713', 'MEG0712', 
          'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1813', 'MEG1812', 'MEG1822', 'MEG1823', 'MEG0743', 'MEG0742',]

roi_right=['MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123', 'MEG1122', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322',
           'MEG0723', 'MEG0722', 'MEG1142', 'MEG1143', 'MEG1133', 'MEG1132', 'MEG1342', 'MEG1343', 'MEG1333', 'MEG1332', 
           'MEG0733', 'MEG0732', 'MEG2212', 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422']


# Make list of variables to save:
beta_peak_freq_l=[]
beta_peak_amp_l=[]
beta_peak_ch_l=[]
beta_pow_total_periodic_l=[]
noise_l=[]
offset_l=[]

beta_peak_freq_r=[]
beta_peak_amp_r=[]
beta_peak_ch_r=[]
beta_pow_total_periodic_r=[]
noise_r=[]
offset_r=[]

Subject=list()
MEG_data_id=list()
Sibling_pair=list()
Age=list()
Gender=list()
Sibling=list()

n_exceptions=12

# Loop for data analysis starts here
for p in range(0, len(df_meta)): 
    subj=df_meta.loc[p]['Number']
    sib=df_meta.loc[p]['Sibling']
    sib_pair=df_meta.loc[p]['Pair']
    age=df_meta.loc[p]['Age']
    gender=df_meta.loc[p]['Gender']
    freq_left=df_meta.loc[p]['freq_left'].split(',')
    freq_right=df_meta.loc[p]['freq_right'].split(',')
    ch_left=df_meta.loc[p]['ch_left'].split(',')
    ch_right=df_meta.loc[p]['ch_right'].split(',')
    force_left=df_meta.loc[p]['force_ch_left']
    force_right=df_meta.loc[p]['force_ch_right']
    for h in range(0, len(txtfiles)-n_exceptions):  
        ###### LOAD RAW DATA
        fn = txtfiles[h]
        fs = fn.split("/")
        tmp=re.split('(\d+)',fs[-2])
        pat_id=int(tmp[1])   
        times=fs[-3].split("_")
        if subj==pat_id:
            raw_eo = sib_load_data_1(fn, fs, times, txtfiles)

            Subject.append(subj) # sujbect and MEG_data_id entries should match in the Pandas file
            MEG_data_id.append(pat_id)
            Sibling_pair.append(sib_pair)
            Age.append(age)
            Gender.append(gender)
            Sibling.append(sib)
                        
            ###### POWER & VECTOR SUM CALCULATION ############################
            sfreq=raw_eo.info['sfreq']
        
            # define subset of channels to use
            picks_left=mne.pick_channels(raw_eo.info['ch_names'], include=roi_left,
                    exclude=[], ordered=True)
            picks_right=mne.pick_channels(raw_eo.info['ch_names'], include=roi_right,
                    exclude=[], ordered=True)
    
            # filtering (fmin, fmax defined earlier)
            raw_eo.filter(fmin, fmax, fir_design='firwin')
            
            # calaculate power spectra for ROI, left
            [power1,f] = mne.time_frequency.psd_welch(raw_eo, picks=picks_left, tmin=None, tmax=None,
                                    fmin=fmin, fmax=fmax, n_fft=n_fft)
    
            # calaculate power spectra for ROI, right
            [power2,f] = mne.time_frequency.psd_welch(raw_eo, picks=picks_right, tmin=None, tmax=None,
                                    fmin=fmin, fmax=fmax, n_fft=n_fft)    
            
            vs_left, labels_left = vector_sum(power1, roi_left)
            vs_right, labels_right = vector_sum(power2, roi_right)
            
            ######## SEPARATE NOISE & PERIODIC COMPONENT ###############
            ######## Use FOOOF to de-trend data (an obtain only periodic component of spectrum)
            PERIODIC_r, NOISE_r, APERIODIC_b_r, APERIODIC_x_r = remove_aperiodic(f, vs_right, freq_range)
            PERIODIC_l, NOISE_l, APERIODIC_b_l, APERIODIC_x_l = remove_aperiodic(f, vs_left, freq_range)

            ###### LOAD MEG CHANNEL SELECTION AND BETA PEAK FREQUENCIES INFO
            ###### FOR FURTHER ANALYSIS
            f_range=1
            def channels_and_frequencies(channel, frequency, f_range):
                if channel[0] == 'n.a.':
                    ch=[]
                    beta_lo=[]
                    beta_hi=[]
                else:
                    ch=[int(x) for x in channel]
                    freq=[int(x) for x in frequency]
                    beta_lo=[x-f_range for x in freq]
                    beta_hi=[x+f_range for x in freq]
                return ch, beta_lo, beta_hi
            
            ch, Beta_lo, Beta_hi = channels_and_frequencies(ch_left, freq_left, f_range)
            ch_right, Beta_lo_right, Beta_hi_right = channels_and_frequencies(ch_right, freq_right, f_range)
            
            # 1st beta peak, first left, then right
            if len(Beta_lo) > 0:
                count=0
                beta_lo=Beta_lo[count]
                beta_hi=Beta_hi[count]
                
                # extract beta information for first set of channel & frequency
                beta_max_freq, beta_max_amp, beta_max_ch, noise_comp, offset_component=find_bmax_info(PERIODIC_l, APERIODIC_x_l, APERIODIC_b_l, f, beta_lo, beta_hi, force_left, count)
                
                # get beta band range total power
                tot_beta, tot_periodic_beta=get_total_beta_power(vs_left, PERIODIC_l, beta_range, beta_max_ch, f)
                            
                beta_peak_ch_l.append(beta_max_ch)   
                beta_peak_freq_l.append(beta_max_freq)
                beta_peak_amp_l.append(beta_max_amp)
                beta_pow_total_periodic_l.append(tot_periodic_beta)
                noise_l.append(noise_comp)
                offset_l.append(offset_component)

            else:
                beta_peak_ch_l.append(float('nan'))   
                beta_peak_freq_l.append(float('nan'))
                beta_peak_amp_l.append(float('nan'))
                beta_pow_total_periodic_l.append(float('nan'))
                noise_l.append(float('nan'))
                offset_l.append(float('nan'))
                
            if len(Beta_lo_right) > 0:
                count=0
                beta_lo=Beta_lo_right[count]
                beta_hi=Beta_hi_right[count]
                
                # extract beta information for first set of channel & frequency
                beta_max_freq_r, beta_max_amp_r, beta_max_ch_r, noise_comp_r, offset_component_r=find_bmax_info(PERIODIC_r, APERIODIC_x_r, APERIODIC_b_r, f, beta_lo, beta_hi, force_right, count)
                
                # get beta band range total power
                tot_beta_r, tot_periodic_beta_r=get_total_beta_power(vs_right, PERIODIC_r, beta_range, beta_max_ch, f)
                
                beta_peak_ch_r.append(beta_max_ch_r)   
                beta_peak_freq_r.append(beta_max_freq_r)
                beta_peak_amp_r.append(beta_max_amp_r)
                beta_pow_total_periodic_r.append(tot_periodic_beta_r)
                noise_r.append(noise_comp_r)  
                offset_r.append(offset_component_r)
                
            else:
                beta_peak_ch_r.append(float('nan'))   
                beta_peak_freq_r.append(float('nan'))
                beta_peak_amp_r.append(float('nan'))
                beta_pow_total_periodic_r.append(float('nan'))
                noise_r.append(float('nan'))  
                offset_r.append(float('nan'))  

# data frame with MEG beta channel & power info
df = pd.DataFrame({'Pair number': Sibling_pair,
                    'Subject number': Subject,
                    'MEG data subject number': MEG_data_id,
                    'Siblings': Sibling,
                    'Age': Age,
                    'Sex': Gender,
                    'peak channel left': beta_peak_ch_l,
                    'peak freq left': beta_peak_freq_l,
                    'peak amp left': beta_peak_amp_l,
                    'beta band power periodic left': beta_pow_total_periodic_l,
                    'aperiodic left': noise_l,
                    'offset left': offset_l,
                    'peak channel right': beta_peak_ch_r,
                    'peak freq right': beta_peak_freq_r,
                    'peak amp right': beta_peak_amp_r,  
                    'beta band power periodic right': beta_pow_total_periodic_r,
                    'aperiodic right': noise_r, 
                    'offset right': offset_r, 
                    }) 

# # save the dataframe with beta power information
df.to_csv(path_metadata + fn_dataframe)
