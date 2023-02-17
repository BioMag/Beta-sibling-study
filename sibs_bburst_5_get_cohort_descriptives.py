#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:00:29 2022

@author: paulsk

Script which generates a PANDAS data frame format list of descriptives 
df_descriptives from the original PSD and beta event phenotype data files 
(specified in df_meta).

The script also checks whether there are missing entries which are also saved 
in a new variable called empty_rows. There should only be one missing row in 
the right hemisphere information (one subject where there was no clear beta peak
                                  on the right).

The descriptives information is saved in a new .csv file beta_descriptives.csv 
in the location specified in path_metadata. The missing entry information is 
also saved in a new .csv file in the same location.

"""

import pandas as pd
import numpy as np

######  PATH INFORMATION 
path_trunc='/main/'
path_metadata=path_trunc + 'metadata_tables/'

# load data
df_meta=pd.read_csv(path_metadata + 'sibling_beta_PSD_and_burst_data_table.csv')

# get column names and store as new variable 
column_names = df_meta.columns.values.tolist()
    
# drop cases which are not included
## case 140 - sibling missing
idx_drop = np.squeeze((np.where(df_meta['Subject number']==140)))
df_meta.drop([df_meta.index[idx_drop]], inplace=True)

## cases 165 & 166 - one sibling had preceeding head trauma so both siblings dropped
idx_drop = np.squeeze((np.where(df_meta['Subject number']==165)))
df_meta.drop([df_meta.index[idx_drop]], inplace=True)

idx_drop3= np.squeeze((np.where(df_meta['Subject number']==166)))
df_meta.drop([df_meta.index[idx_drop]], inplace=True)

df_clean=df_meta

# get descriptives
means=df_clean.mean(axis=0)
medians=df_clean.median(axis=0)
minima=df_clean.min(axis=0)
maxima=df_clean.max(axis=0)
stdevs=df_clean.std(axis=0)

# combine descriptives into new data frame & save
frames = [means, medians, stdevs, minima, maxima]
df_descriptives = pd.concat(frames, axis=1, join='inner')
df_descriptives.to_csv(path_metadata + 'beta_descriptives.csv')

# get empty rows info per column & save
df_clean.isnull().values.ravel().sum()
empty_rows=df_clean.isna().sum()
empty_rows.to_csv(path_metadata + 'beta_missing_entries.csv')


# # some info on interrogating Pandas data frames
# # get column names
# list(df_meta.columns)

# # get one column
# test = df_left['2. peak mean dur']

# # get some entries from particular column
# test = df_left['2. peak mean dur'].loc[200:] # line 200 till end of table

# # check for Nans
# np.isnan(test).any()

