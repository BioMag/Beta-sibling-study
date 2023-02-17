#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:27:04 2021

INPUT :
-------

1. Raw sibling MEG data found under
path_to_data1='/Volumes/LaCie/sibling_data/standard/'
path_to_data2='/Volumes/LaCie/sibling_data/exceptions/'

2. Pandas data frame named 'sibling_bpower.csv', wich contains 
    'Pair number': Sibling_pair,
    'Subject number': Subject,
    'MEG data subject number': MEG_data_id,
    'Siblings': Sibling,
    'Age': Age,
    'Sex': Gender,
    'peak channel left': beta_peak_ch_l,   ##### used further
    'peak freq left': beta_peak_freq_l,   
    'peak amp left': beta_peak_amp_l,
    'aperiodic left': noise_l,
    'peak channel right': beta_peak_ch_r,  ##### used further
    'peak freq right': beta_peak_freq_r,
    'peak amp right': beta_peak_amp_r,   
    'aperiodic right': noise_r

OUTPUT :
-------
Data table which contains the extracted burst data.

"""
import numpy as np
import pandas as pd
import scipy as scipy
import matplotlib.pyplot as plt
import mne
import re

# load custom-written utility functions from file
from sibs_bburst_support_functions import *

######  GENERAL ANALYSIS SETTINGS
fmin, fmax = 2, 48.  # lower and upper band pass frequency for data filtering
side='right' # hemispheres to analyse, options 'right' or 'left'

# Burst analysis
dwn=200 # downsampling target frequency (Hz)
lim = 50  # lower duration limit for beta burst, in ms
percentile=75 # amplitude percentile threshold

######  PATH INFORMATION 
path_trunc='/main/'
path_to_data1=path_trunc + 'raw_data/standard/'
path_to_data2=path_trunc + 'raw_data/exceptions/'
path_to_figs=path_trunc + 'figures/'
path_metadata=path_trunc + 'metadata_tables/'

###### FILE NAMES FOR SAVING
fn_dataframe='sibling_bburst_%s.csv' % side

######  LOAD SUBJECT METADATA
df_meta=pd.read_csv(path_metadata +'sibling_beta_PSD_phenotype.csv')

######  MEG DATA FILENAME INFORMATION & PATHS
import glob

txtfiles=[]
folders1=glob.glob(path_to_data1+'*')
for i in range(0,len(folders1)):
    tmp=glob.glob(folders1[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)    
folders1=glob.glob(path_to_data2+'0_180_190_360_0_0/')
for i in range(0,len(folders1)):
    tmp=glob.glob(folders1[i] + '/nro*/*sss*.fif')
    txtfiles.append(tmp)   
txtfiles=np.concatenate(txtfiles)

# define ROI
if side=='left':
    ROI=['MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0633', 'MEG0632',
              'MEG0243', 'MEG0242', 'MEG0232', 'MEG0233', 'MEG0443', 'MEG0442', 'MEG0432', 'MEG0433', 'MEG0713', 'MEG0712', 
              'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1813', 'MEG1812', 'MEG1822', 'MEG1823', 'MEG0743', 'MEG0742',]
else:
    ROI=['MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123', 'MEG1122', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322',
               'MEG0723', 'MEG0722', 'MEG1142', 'MEG1143', 'MEG1133', 'MEG1132', 'MEG1342', 'MEG1343', 'MEG1333', 'MEG1332', 
               'MEG0733', 'MEG0732', 'MEG2212', 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422']
        
        
# Make list of variables to save:
Subject=list()
MEG_data_id=list()
Sibling_pair=list()
Sibling=list()
Age=list()
Gender=list()
bdur_mean=list()
bdur_median=list()
bdur_std=list()
bdur_robustmax=list()
bamp_mean=list()
bamp_median=list()
bamp_std=list()
bamp_robustmax=list()
bpersecond=list()
bdispersion=list()

# Loop for data analysis starts here
for p in range(0, len(df_meta)): 
    subj=df_meta.loc[p]['Subject number']
    sib=df_meta.loc[p]['Siblings']
    sib_pair=df_meta.loc[p]['Pair number']
    age=df_meta.loc[p]['Age']
    gender=df_meta.loc[p]['Sex']
    freq_left=df_meta.loc[p]['peak freq left']
    freq_right=df_meta.loc[p]['peak freq right']
    ch_left=df_meta.loc[p]['peak channel left']
    ch_right=df_meta.loc[p]['peak channel right']
    amp_left=df_meta.loc[p]['peak amp left']
    amp_right=df_meta.loc[p]['peak amp right']

    for h in range(0, len(txtfiles)):  
        ###### LOAD RAW DATA
        fn = txtfiles[h]
        fs = fn.split("/")
        tmp=re.split('(\d+)',fs[-2])
        pat_id=int(tmp[1])   
        times=fs[-3].split("_")
        
        if subj==pat_id:
            raw_eo = sib_load_data_1(fn, fs, times, txtfiles)
            Subject.append(subj)
            MEG_data_id.append(pat_id)
            Sibling_pair.append(sib_pair)
            Age.append(age)
            Gender.append(gender)
            Sibling.append(sib)
            
            if side=='left':
                channel_1=ch_left
                frequency_1=freq_left
            else:
                channel_1=ch_right
                frequency_1=freq_right
                
            if np.isnan(channel_1) == True:
                bdur_mean.append(float('nan'))
                bdur_median.append(float('nan'))
                bdur_std.append(float('nan'))
                bdur_robustmax.append(float('nan'))
                bamp_mean.append(float('nan'))
                bamp_median.append(float('nan'))
                bamp_std.append(float('nan'))
                bamp_robustmax.append(float('nan'))                 
                bpersecond.append(float('nan'))
                bdispersion.append(float('nan'))
            else:
                dmean1, dmedian1, dstdev1, dmax1, drobustmax1, dmean2, dmedian2, dstdev2, dmax2, drobustmax2, bps, di = get_burst_info(int(channel_1), frequency_1, ROI, raw_eo, percentile, lim, dwn, fmin)
                    
                bdur_mean.append(dmean2)
                bdur_median.append(dmedian2)
                bdur_std.append(dstdev2)
                bdur_robustmax.append(drobustmax2)
                bamp_mean.append(dmean1)
                bamp_median.append(dmedian1)
                bamp_std.append(dstdev1)
                bamp_robustmax.append(drobustmax1)                 
                bpersecond.append(bps)
                bdispersion.append(di)      
                
# data frame with MEG beta bursting info
df = pd.DataFrame({'Subject number': Subject,
                    'MEG data subject number': MEG_data_id,
                    'Pair number': Sibling_pair,
                    'Siblings': Sibling,
                    'Age': Age,
                    'Sex': Gender,
                    'mean dur': bdur_mean,
                    'median dur': bdur_median,
                    'std dur': bdur_std,
                    'robustmax dur': bdur_robustmax,
                    'mean amp': bamp_mean,
                    'median amp': bamp_median,
                    'std amp': bamp_std,
                    'robustmax amp': bamp_robustmax,
                    'bps': bpersecond,                   
                    'dispersion': bdispersion
                    })

# sort by ascending subject number
df.sort_values('Subject number', axis=0, ascending=True)

# saving the dataframe
df.to_csv(path_metadata + fn_dataframe)
     
# read dataframe
# # test=pd.read_csv('sibling_bburst_right.csv')
