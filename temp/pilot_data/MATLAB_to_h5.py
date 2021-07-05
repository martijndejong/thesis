# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:48:09 2020

@author: Martijn de Jong

This script will do the following:
    - Read all MATLAB .csv generated pilot data
    - Add columns: 'e_dot', 'u_dot', and 'class'
    - Standardise data
    - Filter to only include successful tracking runs
    - Combine above data into one .h5 file
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# time derivative function
def derivative(df, column, dt=0.01):
    y = np.array(df[column])
    y_dot = np.zeros(len(y))
    for i in range(len(y)):
        if i == 0:
            y_dot[i] = (y[i+1] - y[i])/dt       # first order forward difference approximation
        elif i == (len(y)-1):
            y_dot[i] = (y[i] - y[i-1])/dt       # first order backward difference approximation
        else:
            y_dot[i] = (y[i+1] - y[i-1])/(2*dt) # second-order centered difference approximation
        
    return y_dot

# standardise function
def standardise(df, column):
    y = df[column]
    y_mean = np.mean(y) # mean
    y_std = np.std(y)   # standard deviation
    y = (y - y_mean)/y_std 
    
    return y


# read in SubjectRuns folder to attain additional information
def read_info():
    root = './real/SubjectRuns' # set root of file locations
    data = pd.DataFrame() # empty dataframe to concatenate individual data files to
    
    files = os.listdir(root)
    
    print("Reading SubjectRuns info")
    time.sleep(0.5)
    
    for file in tqdm(files):
        # take information from file name
        info = file.split('_')
        group   = int(info[1].split('Group')[1])
        subject = int(info[2].split('Subject')[1].split('.dat')[0])
        
        # select what fields to read
        fields = ['run #:', 'RMS e:', 'RMS u:']    
        # read in data from .dat file
        df = pd.read_csv(os.path.join(root,file), 
                                     sep = '\s\s+|,', engine='python', 
                                     header=5, usecols=fields).drop(0)
        # fix formatting
        df.rename(columns={'run #:':'run', 'RMS e:':'RMS e', 'RMS u:':'RMS u'}, inplace = True) # rename columns
        
        # add info columns
        df['group']     = group
        df['subject']   = subject
        df['id']    = df['subject'].astype(str) + '|' + df['run'].astype(str)
        
        # add to total data set
        data = pd.concat([data, df])
    # only save 'No motion' group info
    data = data[data['group']==1]
    # data.to_hdf('subject_runs.h5', key='info')           
    return data


# read, format, configure, and concatenate pilot data to .h5
def read_data():
    root = './generated' # set root of file location
    
    data = pd.DataFrame() # empty dataframe to concatenate individual data files to
    
    files = os.listdir(root) # list of all files per data folder

    print("Reading generated pilot data with actual parameters")
    time.sleep(0.5)
    for file in tqdm(files):
        # take information from file name
        info    = file.split('_')
        group   = int(info[1].split('Group')[1])
        run     = int(info[3].split('nr')[1])
        subject = int(info[2].split('Subject')[1])
        motion  = 0 if info[4] == 'motOFF' else 1  # 0 = OFF / 1 = ON
        
        # READING FILTERS
        run_bool = True if run<=20 or run>=80 else False # only read first and last 20 runs
        grp_bool = True if group == 1 else False         # only read group 1
        mot_bool = True if motion == 0 else False        # only read motion off
        
        if grp_bool and mot_bool and run_bool: 
            fields = ['t', 'ft', 'fd', 'e', 'DYN u', 'PCTRLS uy'] #specify what columns to read
            df = pd.read_csv(os.path.join(root,file), 
                             sep = '\s\s+|,', engine='python', 
                             usecols=fields).drop(0)
            
            # fix formatting
            df['t']         = round(df['t'].astype(float),2) # convert time to float data type
            df.rename(columns={'PCTRLS uy':'u'}, inplace = True) # rename column u 
            df['e'] = df['e']*(180/np.pi) # CONVERT TO DEGREES
            df['u'] = df['u']*(180/np.pi) # CONVERT TO DEGREES  
            
            # add info columns
            df['group']     = group
            df['subject']   = subject
            df['run']       = run
            df['motion']    = motion
            
            # add columns (derivative, class, and id)
            df['edot']  = derivative(df, 'e')
            df['udot']  = derivative(df, 'u')
            df['class'] = np.where(df['run']>=50, 1, 0) #split in the middle i.e. class=0 if run<50, else class=1
            df['id']    = df['subject'].astype(str) + '|' + df['run'].astype(str)
            
            # slice time. Remove first 8.08s and last 5s
            # time info:
                # run-in time: 8.08
                # measurement time: 81.92
                # total time: 95
            df = df[(df['t']>=8.08) & (df['t']<=(90))] 
            
            #  make copies of original values before standardising
            df['e_copy']    = df['e']      
            df['edot_copy'] = df['edot']
            df['u_copy']    = df['u']
            df['udot_copy'] = df['udot']
            
            # standardise per run
            columns = ['e', 'u', 'edot', 'udot']
            for column in columns:
                df[column]     = standardise(df, column)
                
                                    
            # # downsample to 50hz (from 100hz) --- removed feature
            # df = df.iloc[::2]
            # df = df.reset_index(drop=True)
            
                           
            # add to total data set
            data = pd.concat([data, df])
    
    # using dictionary to convert specific columns 
    convert_dict = {'group':        'int8', 
                    'subject':      'int8',
                    'run':          'int16',
                    'motion':       'int8',
                    'class':        'int8',
                    'id':            str,
                    
                    'ft':           'float32',
                    'fd':           'float32',
                    'e':            'float32',
                    'edot':         'float32',
                    'u':            'float32',
                    'udot':         'float32',
                    
                    'e_copy':       'float32',
                    'edot_copy':    'float32',
                    'u_copy':       'float32',
                    'udot_copy':    'float32'
                    } 
    
    data = data.astype(convert_dict)
    # data.to_hdf('actual_pilot_data.h5', key='test')           
    return data


# only use data that has information in 'SubjectRuns'
def filter_info(info, data):
    ids = info['id'].unique()
    data = data[data['id'].isin(ids)]
      
    return data            


# ---------- run functions ----------------

info = read_info() # retrieve additional info about runs from SubjectRuns files
 
data = read_data()  # run this to put simona data into dataframe and save to h5

data = filter_info(info, data) # filter out runs that are not present in SubjectRuns

data.to_hdf('simulated_pilot_data.h5', key='complete') # save collected data

