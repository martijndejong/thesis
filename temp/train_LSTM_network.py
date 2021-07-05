# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:01:26 2021

@author: Martijn de Jong
"""

#------------------------------PREPARING DATA----------------------------------
"""
This part of the code concerns applying the desired data settings
"""

# import packages
import pandas as pd
import numpy as np

# import custom input data settings functions
import input_data_functions as func

import random

desc = 'p2p' # description used in log files

widths = [1, 5, 10, 15, 20, 25]
network = 'resnet'

for width in widths:

    seeds = list(range(1,21))
    
    for seed in seeds:
        np.random.seed(seed) # set a random seed
        random.seed(seed)
        
        filename = './pilot_data/actual_pilot_data.h5'
        samplingfreq = 50 # hz
        windowsize = 1.6 # seconds 1.6
        OL = 0.9 # overlap
        input_var = ['e', 'edot', 'u', 'udot'] # selected variables to train network
        
        label_method    = 'run' # method can be 'run'/'RMSe'/'RMSu'
        label_runs      = width    # number of labeled runs per class per subject (and used for training)
        
        
        # read .h5 file containing all the pilot tracking data 
        data = pd.read_hdf(filename, key = 'complete') 
        
        # label the data
        data = func.label_data(data, method=label_method, runs=label_runs)
        
        # split the data into train/test
        train_data, test_data = func.split_data(data, VAL_PCT=0.2)
        
        # draw (overlapping) samples from data and shape into required dimensions for LSTM model
        print("\nHandling training data")
        train_X, train_y = func.format_data(train_data, overlap=OL, SF=samplingfreq, 
                                            WS=windowsize, variables=input_var) # format train data, 90% overlap
        
        print("\nHandling testing data")
        test_X, test_y = func.format_data(test_data, overlap=0.2, SF=samplingfreq, 
                                          WS=windowsize, variables=input_var) # format test data, 20% overlap
        
        # change numpy array data type to float32
        train_X = train_X.astype(np.float16)
        train_y = train_y.astype(np.float16)
        test_X = test_X.astype(np.float16)
        test_y = test_y.astype(np.float16)   
        
        del data, train_data, test_data # clear unused memory space
        #------------------------------BUILDING MODEL----------------------------------
        """
        This part of the code concerns setting up the LSTM model and training it
        """
        
        
        from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
        
        
        seq_length = int(samplingfreq*windowsize)
        num_var = len(input_var)
        
        # # --------------------LSTM---------------------------
        
        # from keras.models import Sequential #, load_model
        # from keras.layers import Dense, Activation, Dropout, LSTM
        # # Model architecture
        # hidden_neurons = 100 # 100
        # dropout = 0.2
        # model = Sequential()
        # model.add(LSTM(hidden_neurons, input_shape=(seq_length, num_var), return_sequences=True))
        # model.add(Dropout(dropout))
        # model.add(LSTM(hidden_neurons))
        # model.add(Dropout(dropout))
        # model.add(Dense(2))
        # model.add(Activation("softmax"))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        
        # # --------------------LSTM---------------------------
        
        
        
        
        # Callback function to save checkpoints
        checkpoint_path = "./log/train_checkpoints/"+desc+'_'\
            +'OL'+str(OL)+'_'\
            +'SF'+str(samplingfreq)+'_'\
            +'WS'+str(windowsize)+'_'\
            +'VAR'+'-'.join(input_var)+'_'\
            +"-ep-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.hdf5"
                                                      
        cp_callback = ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=False,
            save_best_only=True,
            monitor = 'val_loss')
        
        
        
        if network == 'incept':
            # inception-------------
            from inception import Classifier_INCEPTION 
            
            model = Classifier_INCEPTION(output_directory='./temp', input_shape=(seq_length, num_var), nb_classes=2, depth=4, use_bottleneck=False)
            model.model.summary()
            
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.00005) #min_lr=0.0001
            desc = 'incept-nobot'
            # inception-------------
        
        
        if network =='fcn':
            # fullyconvolutional-------------
            from fcn import Classifier_FCN
            
            model = Classifier_FCN(output_directory='./temp/', input_shape=(seq_length, num_var), nb_classes=2)
            model.model.summary()
            
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.00005) #min_lr=0.0001
            desc = 'fcn'
            # fullyconvolutional-------------
        
        
        if network == 'resnet':
            # residualnetwork-------------
            from resnet import Classifier_RESNET
            
            model = Classifier_RESNET(output_directory='./temp/', input_shape=(seq_length, num_var), nb_classes=2)
            model.model.summary()
            
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.00005) #min_lr=0.0001
            desc = 'resnet'
            # residualnetwork-------------
        
        
        
        
        
        # Training network
        epochs = 20
        batch_size  = 64 # nvidia nsight set TDR 10 sec
        
        hist = model.model.fit(train_X, #model.model.fit
                         train_y,
                         batch_size = batch_size,
                         epochs = epochs,
                         validation_data = (test_X, test_y),
                         shuffle = True,
                         callbacks=[cp_callback, reduce_lr])
        
        train_loss  = hist.history['loss']
        train_acc   = np.array(hist.history['accuracy'])*100
        val_loss    = hist.history['val_loss']
        val_acc     = np.array(hist.history['val_accuracy'])*100
        
        
        
        
        # # ------------------------- Plotting results --------------------------------
        # import matplotlib.pyplot as plt
        
        # fig = plt.figure()
        # fig.set_size_inches(7.5,6)
        
        # cyan = 'deepskyblue'
        # red = 'r'
        
        # ax1 = plt.subplot2grid((2, 1), (0, 0))
        # ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1) 
        
        # ax1.plot(range(epochs), train_acc, label="Training accuracy", c=red, linewidth=2.5)
        # ax1.plot(range(epochs), val_acc, label="Validation accuracy", c=cyan, linewidth=2.5)
        # ax1.legend()
        # ax1.set_ylabel('Accuracy [%]')
        
        # ax2.plot(range(epochs), train_loss, label="Training loss", c=red, linewidth=2.5)
        # ax2.plot(range(epochs), val_loss, label="Validation loss", c=cyan, linewidth=2.5)
        # ax2.legend()
        # ax2.set_ylabel('Loss [-]')
        # ax2.set_xlabel('Epoch')
        
        # plt.show()
        
        
        
        
        #--------------------saving training information to log-----------------------
        log = {'train_acc':     train_acc,
               'val_acc':       val_acc,
               'train_loss':    train_loss,
               'val_loss':      val_loss}
        
        log = pd.DataFrame(log)
        
        best_acc = round(float(log.val_acc[log.val_loss==log.val_loss.min()]),3)
        best_loss = round(float(log.val_loss[log.val_loss==log.val_loss.min()]),3)
        
        # log_path = "./log/train_log/"+desc+'_'\
        #     +'OL'+str(OL)+'_'\
        #     +'SF'+str(samplingfreq)+'_'\
        #     +'WS'+str(windowsize)+'_'\
        #     +'VAR'+'-'.join(input_var)\
        #     +'.h5'
        
        log_path = "./log/pm15/"+desc+'_'\
                +'s'+str(seed)+'_'\
                +'w'+str(width)+'_'\
                +'acc-'+str(best_acc)+'-loss-'+str(best_loss)+'.h5'
                    
                    
                # +'acc-{best_acc:.3f}-loss-{best_loss:.3f}.h5'
            
        log.to_hdf(log_path, key='log')  
        
    




