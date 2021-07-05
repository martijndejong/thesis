# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:13:30 2021

@author: Martijn de Jong

This code defines the following data handling functions:
    standardise()   - standardise data
    label_data()    - add class labels to runs
    split_data()    - split data into train/test set with 50/50 distribution of classes
    format_data()   - format the tracking runs into overlapping samples of specified WS and SF
    shuffle()       - shuffle samples while maintaining correct index (Xi, Yi) 
"""
# import packages
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# standardise function definition
def standardise(df, column):
    y = df[column]
    y_mean = np.mean(y) # mean
    y_std = np.std(y)   # standard deviation
    y = (y - y_mean)/y_std 
    
    return y


# add class label to data
# 'data' must be a pandas dataframe
# 'method' determines what the class label is based on
# 'runs' dictates what part of the data is used. E.g. runs=10 means the first/last 10 tracking runs of each subject will be used
def label_data(data, method='run', runs=10): 
    data['class'] = np.nan # remove default class from dataframe 
    
    # some runs were inproper and removed from data set, thus index of first/last 10 runs will be different for each subject
    subjects = data['subject'].unique() # list of all subjects in data
    for subject in subjects:
        run_ids = np.unique(data.loc[data.subject==subject, 'run']) # list of run ids for subject
        lower_l   = np.max(run_ids[:runs]) # lower limit, the last run id of the first 10 runs
        upper_l   = np.min(run_ids[-runs:]) # upper limit, the first run id of the last 10 runs
        
        # assign classes to each subject based on the run index (method='run')
        data.loc[(data['run']<=lower_l) & (data['subject']==subject), 'class'] = 0
        data.loc[(data['run']>=upper_l) & (data['subject']==subject), 'class'] = 1
        
    # remove rows without label    
    data.dropna(subset = ["class"], inplace=True)
    
    # run method means that the class label is solely based on the run index of each tracking run
    if method == 'run':
        print('Class label is based on run index')
    
    # RMSe method means that the class label is determined by the RMSe of each tracking run
    if method == 'RMSe':
        print('Class label is based on RootMeanSquared tracking error')
        data.loc[data['RMSe']>=np.median(data['RMSe']), 'class'] = 0 # if RMSe value is larger than the median value: pilot is labeled uskilled
        data.loc[data['RMSe']<=np.median(data['RMSe']), 'class'] = 1 # if RMSe value is smaller than the median value: pilot is labeled skilled
           
    # RMSu method means that the class label is determined by the RMSu of each tracking run
    if method == 'RMSu':
        print('Class label is based on RootMeanSquared pilot output')
        data.loc[data['RMSu']<=np.median(data['RMSu']), 'class'] = 0 # if RMSu value is smaller than the median value: pilot is labeled uskilled
        data.loc[data['RMSu']>=np.median(data['RMSu']), 'class'] = 1 # if RMSu value is larger than the median value: pilot is labeled skilled
        
    # make sure class is stored as integer and return    
    data = data.astype({'class': 'int8'})
    return data

# split data into train and test set;
# each tracking run is split into two halves: A, and B
# each half of every tracking run will become either 'train' or 'test'
# (per half tracking run to get better shuffling distribution than making entire tracking runs 'train' or 'test')
# splitting (half) tracking runs BEFORE overlapping/shuffling to prevent overlap between train&test
def split_data(data, VAL_PCT=0.2):
    print("Splitting data; Training: %2d%%  Testing: %2d%%" % ((1-VAL_PCT)*100, VAL_PCT*100))
    
    # double the amount of run ids by splitting them in two
    data.loc[data.t<49.04, 'id']    += '|A' # every first half of every run is marked with A
    data.loc[data.t>=49.04, 'id']   += '|B' # every first half of every run is marked with A
    
    # generate list of unique ids (no duplicates)
    class0_ids  = data.id[data['class']==0].drop_duplicates() # list of ids belonging to class 0; unskilled
    class1_ids  = data.id[data['class']==1].drop_duplicates() # list of ids belonging to class 1; skilled
    all_ids     = data.id.drop_duplicates()                   # list with all ids
    
    # split ids in test and train
    test_ids    = list(pd.concat([class0_ids.sample(frac=VAL_PCT),    # randomly draw 20% of class 0 ids
                                  class1_ids.sample(frac=VAL_PCT)]))  # randomly draw 20% of class 1 ids
    train_ids   = list(all_ids[~all_ids.isin(test_ids)])              # use remaining ids as train ids
    
    # split data into test and train
    test_data   = data[data.id.isin(test_ids)]  # filter the data to test_data based on marked id
    train_data  = data[data.id.isin(train_ids)] # filter the data to train_data based on marked id
        
    
    print("   ---Distribution of all data---")
    class0 = len(data['id'][data['class']==0].unique())
    class1 = len(data['id'][data['class']==1].unique())
    nruns  = len(data['id'].unique())  
    print("   Total runs:", nruns)
    print("   Unskilled:", class0, ",", round(np.divide(class0,nruns)*100,2), "%")
    print("     Skilled:", class1, ",", round(np.divide(class1,nruns)*100,2), "%")    
    
    print("   ---Distribution training data---")
    class0 = len(train_data['id'][train_data['class']==0].unique())
    class1 = len(train_data['id'][train_data['class']==1].unique())
    nruns  = len(train_data['id'].unique())    
    print("   Total runs:", nruns)
    print("   Unskilled:", class0, ",", round(np.divide(class0,nruns)*100,2), "%")
    print("     Skilled:", class1, ",", round(np.divide(class1,nruns)*100,2), "%")
    
    print("   ---Distribution testing data---")
    class0 = len(test_data['id'][test_data['class']==0].unique())
    class1 = len(test_data['id'][test_data['class']==1].unique())
    nruns  = len(test_data['id'].unique())   
    print("   Total runs:", nruns)
    print("   Unskilled:", class0, ",", round(np.divide(class0,nruns)*100,2), "%")
    print("     Skilled:", class1, ",", round(np.divide(class1,nruns)*100,2), "%")
    
    
    return train_data, test_data


# Formatting the loaded data into the desired shape and overlap
# desired shape: (#samples, sequence length, #variables)
def format_data(data, overlap=0, SF=50, WS=1.6, variables=['e','edot','u','udot']):
    print("Formatting data. Overlap=", overlap*100, "%")
    data = data[variables+['t', 'id', 'class']]
    
    time.sleep(0.3) # wait before tqdm() is called
    
    # First downsample to desired sampling frequency
    dt = 1/SF
    t_list = np.round(np.arange(np.min(data['t']), np.max(data['t'])+dt, dt),2)
    data = data[data['t'].isin(t_list)]
    
    
    # step size of lower limit depends on overlap.  int(round()) to prevent roundown error
    step_size = int(round(SF*WS*(1-overlap),3))
    # window size, number of data points in every sample
    window_size = int(round(SF*WS,3))
    
    # get list of unique identifiers (subject|run) 
    ids = data['id'].unique()
    
    # create empty lists to store sequences and labels
    xs=[]
    ys=[]
    
    # group dataframe by run id, to be able to draw separate runs more quickly
    grouped = data.groupby(data.id)
    
    tic = time.time()
    # split data per unique id (subject|run) so that a sequence can never overlap between different subjects/runs
    for run in tqdm(ids): 
        #filter out run
        run_df = grouped.get_group(run) # much faster than run_df = data['id'==run]
        
        # collect time traces and class of selected run
        run_data = np.array(run_df[variables])
        class_label = np.array(run_df['class'])[0]        
            
        
        # compute matrix with indices to collect overlapping samples
        max_time = len(run_data)-window_size            
        sub_windows = (
        np.expand_dims(np.arange(window_size), 0) +
        # Create a rightmost vector as [0, V, 2V, ...].
        np.expand_dims(np.arange(max_time + 1, step=step_size), 0).T
        )
        
        # extract samples from run data
        x_run = run_data[sub_windows]
        # add corresponding one-hot class label
        y_run = np.zeros((len(x_run),2))
        y_run[:,class_label] = 1
        
        # append to list of runs
        xs.append(x_run)
        ys.append(y_run)    
        
    tac = time.time()
    print('Time spent: ', round((tac-tic),2), 'seconds')
    
    # concatenate samples from different runs into one long array of samples
    X = np.concatenate(xs)
    y= np.concatenate(ys)
    
        
    return X, y.astype(np.int8)








def format_data_OLD(data, overlap=0, SF=50, WS=1.6, variables=['e','edot','u','udot']):
    print("Formatting data. Overlap=", overlap*100, "%")
    data = data[variables+['t', 'id', 'class']]
    time.sleep(0.5) # wait before tqdm() is called
    
    # First downsample to desired sampling frequency
    dt = 1/SF
    t_list = np.round(np.arange(np.min(data['t']), np.max(data['t'])+dt, dt),2)
    data = data[data['t'].isin(t_list)]
    
    # create empty lists to store sequences and labels
    xs=[]
    ys=[]
    
    # step size of lower limit depends on overlap.  int(round()) to prevent roundown error
    step_size = int(round(SF*WS*(1-overlap),3))
    
    # get list of unique identifiers (subject|run) 
    ids = data['id'].unique()
    
    tic = time.time()
    # split data per unique id (subject|run) so that a sequence can never overlap between different subjects/runs
    for run in tqdm(ids): 
        run_data = data[data['id']==run]
        
        # #standardise per run
        # for var in variables:
        #     data[var] = standardise(data, var)
    
        # cycle through run data, increment of lower limit is determined by step_size
        for i in range(0, len(run_data), step_size):
            if i+int(SF*WS) > len(run_data): # break if upper limit is out of bound (i.e. cannot make desired sequence length with remaining data)
                # print("Could not make range:", i,'-', i+int(samplingfreq*frame), "Length run data:",len(run_data)) # turn on this print to see upper limit out of bound
                break
            # print("range:", i,'-', i+int(samplingfreq*frame)) # turn on this print statement to see lower limit and upper limit
            
            # filter out one sequence
            sub_data = run_data[i:int(i+SF*WS)].reset_index(drop=True)
            if len(sub_data['class'].unique()) > 1:
                print("---SEQUENCE CONTAINS TWO CLASSES---")
            
            # append sequence for each variable
            s_seq_lst = []
            for var in variables:
                s_seq = sub_data[var]
                s_seq_lst.append(s_seq)        
            
            # create list of sequences
            xs.append(s_seq_lst)
            # create list of labels
            ys.append(np.eye(2)[sub_data['class'][0]])    

    tac = time.time()
    print('Time spent: ', round((tac-tic),2), 'seconds')
    
    # convert X and y to np.arrays and switch order of axes for X
    X = np.reshape(xs,(-1,int(SF*WS),len(variables))) # just switching order of axis, seq_len <-> #variables
    y = np.array(ys)
    
    return X, y # X.astype(np.float64), y.astype(np.int64)


# shuffle data per sequence
def shuffle(x, y):
    print("Shuffling data")
    indices = np.array(range(0, x.shape[0])) # create array with all indices
    np.random.shuffle(indices) # set indices in random order
    
    x_shuffled = x[indices] # arange data in same random order
    y_shuffled = y[indices] # arange data in same random order

    return x_shuffled, y_shuffled