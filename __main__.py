# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:21:17 2021

@author: Martijn de Jong
"""
# imports
import pandas as pd
import numpy as np
import random
import os

# import custom input data settings functions
import input_data_functions as func


seeds = range(1,31)
loos = None # range(1,14)

split = 'random' if seeds != None else 'loo' # 'random'/'loo' (Leave One Out)

# seed = None
loo = None

for seed in seeds:
    
    # ----------------------------- user input -----------------------------------
    
    # Data settings
    filename = 'data/all_actual_pilot_data.h5'
    samplingfreq = 50 # Hz
    windowsize = 1.2 # s
    overlap = 0.9 # %
    label_width = 20 # number of runs for each participant that are labeled as skilled or unskilled
    label_method = 'run' # 'run'/'RMSe'/'RMSu', method to determine class label of tracking run
    variables = ['e', 'edot', 'u', 'udot'] # selected input variables to train network
    seq_length = int(samplingfreq*windowsize) # length (number of time steps) of one sample fed to ANN
    num_var = len(variables) # number of variables
    
    # Network settings
    name = 'resnet'
    epochs = 20
    batch_size = 128
    lr = 0.0005 # learning rate of network (alpha) ~standard = 0.001
    min_lr = 0.00005 # minimum learning rate ~standard = 0.0001
    
    # ----------------------------------------------------------------------------
    
    # set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # compute output directory name
    directory = name+ '_L-'+label_method+\
                '_W-'+str(label_width)+\
                '_OL-'+str(overlap)+\
                '_SF-'+str(samplingfreq)+\
                '_WS-'+str(windowsize)+\
                '_V-'+'-'.join(variables)+\
                '_B-'+str(batch_size)+\
                '_LR-'+str(lr)+\
                '_MLR-'+str(min_lr)
    
    current = 'seed-'+str(seed) if split=='random' else 'loo-'+str(loo) # name of current working folder      
    directory = os.path.join('log', directory, current)+'\\' # name of full path to output directory
    os.makedirs(directory, exist_ok=True) # create full path to output directory
                
    
    # compute data
    # read .h5 file containing all the pilot tracking data 
    data = pd.read_hdf(filename, key = 'complete') 
    
    # label the data
    data = func.label_data(data, method=label_method, runs=label_width)
    
    # split the data into train/test
    train_data, test_data = func.split_data(data, VAL_PCT=0.2, output_directory=directory, method=split, loo=loo)
    
    # draw (overlapping) samples from data and shape into required dimensions
    print("\nHandling training data")
    train_X, train_y = func.format_data(train_data, overlap=overlap, SF=samplingfreq, 
                                        WS=windowsize, variables=variables) # format train data, 90% overlap
    
    print("\nHandling testing data")
    test_X, test_y = func.format_data(test_data, overlap=0.2, SF=samplingfreq, 
                                      WS=windowsize, variables=variables) # format test data, 20% overlap
    
    del data, train_data, test_data # clear unused memory space
    
    
    # create and train network
    if name == 'resnet':
        from networks.resnet import Classifier_RESNET         
        model = Classifier_RESNET(output_directory=directory, input_shape=(seq_length, num_var), 
                                     nb_classes=2, lr=lr, min_lr=min_lr,
                                     nb_epochs=epochs, batch_size=batch_size)

    if name == 'fcn':
        from networks.fcn import Classifier_FCN         
        model = Classifier_FCN(output_directory=directory, input_shape=(seq_length, num_var), 
                                     nb_classes=2, lr=lr, min_lr=min_lr,
                                     nb_epochs=epochs, batch_size=batch_size)

    if name == 'inception':
        from networks.inception import Classifier_INCEPTION         
        model = Classifier_INCEPTION(output_directory=directory, input_shape=(seq_length, num_var), 
                                     nb_classes=2, lr=lr, min_lr=min_lr,
                                     nb_epochs=epochs, batch_size=batch_size)

    if name == 'lstm':
        from networks.lstm import Classifier_LSTM         
        model = Classifier_LSTM(output_directory=directory, input_shape=(seq_length, num_var), 
                                     nb_classes=2, lr=lr, min_lr=min_lr,
                                     nb_epochs=epochs, batch_size=batch_size)        
    
    if name == 'resnet2d':
        train_X = np.expand_dims(train_X, axis=3)
        test_X = np.expand_dims(test_X, axis=3)
        from networks.resnet_2d import Classifier_RESNET2D        
        model = Classifier_RESNET2D(output_directory=directory, input_shape=(seq_length, num_var, 1), 
                                     nb_classes=2, lr=lr, min_lr=min_lr,
                                     nb_epochs=epochs, batch_size=batch_size)
        
    
    hist, best_acc, best_loss, best_ep = model.fit(x_train=train_X, y_train=train_y, x_test=test_X, y_test=test_y)
    
    
    os.rename(directory, directory[:-1]+'_ep-'+str(best_ep)+'_acc-'+str(round(best_acc,3))+'_loss-'+str(round(best_loss,3)))
    
    
