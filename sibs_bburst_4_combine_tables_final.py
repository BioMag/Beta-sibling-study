#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:32:15 2022

@author: amande
"""
import pandas as pd
import numpy as np

######  PATH INFORMATION 
path_trunc='/main/'
path_metadata=path_trunc + 'metadata_tables/'

###### FILES TO LOAD
fn_left='sibling_bburst_left.csv' 
fn_right='sibling_bburst_right.csv' 
fn_meta='sibling_beta_PSD_phenotype.csv'

###### FILE NAMES FOR SAVING
fn_dataframe='sibling_beta_PSD_and_burst_data_table.csv'

# load tables
df_left=pd.read_csv(path_metadata + fn_left)
df_right=pd.read_csv(path_metadata + fn_right)
df_meta=pd.read_csv(path_metadata + fn_meta)

# rename columns (left & right)
df_left.rename(columns={'mean dur': 'mean dur left',
                        'median dur': 'median dur left',
                        'std dur': 'std dur left',
                        'robustmax dur': 'robustmax dur left',
                        'mean amp': 'mean amp left',
                        'median amp': 'median amp left',
                        'std amp': 'std amp left',
                        'robustmax amp': 'robustmax amp left',
                        'bps': 'bps left',
                        'dispersion': 'dispersion left',
                        }, inplace=True)

df_right.rename(columns={'mean dur': 'mean dur right',
                        'median dur': 'median dur right',
                        'std dur': 'std dur right',
                        'robustmax dur': 'robustmax dur right',
                        'mean amp': 'mean amp right',
                        'median amp': 'median amp right',
                        'std amp': 'std amp right',
                        'robustmax amp': 'robustmax amp right',
                        'bps': 'bps right',
                        'dispersion': 'dispersion right',
                        }, inplace=True)

# drop unnecessary columns
df_left.drop(columns=['Unnamed: 0'])
df_right.drop(columns=['Unnamed: 0'], inplace=True)
df_meta.drop(columns=['peak channel left', 'peak channel right'], inplace=True)

# combine left & right table
df_burst=pd.merge(df_left, df_right, on=["Subject number", "MEG data subject number", "Pair number",  "Siblings",  "Age", "Sex"])
df=pd.merge(df_meta, df_burst, on=["Subject number", "MEG data subject number", "Pair number",  "Siblings",  "Age", "Sex"])
df.drop(columns= ['Unnamed: 0_y','Unnamed: 0_x'], inplace=True)

# saving the dataframe
df.to_csv(path_metadata + fn_dataframe)

    