# -*- coding: utf-8 -*-
"""
Original code by Ismail Fawaz

Additions by Martijn de Jong
"""

import tensorflow.keras as keras
import numpy as np
import pandas as pd
import time


class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True, build=True,
                 batch_size=64, nb_epochs=30, lr=0.001, min_lr=0.0001,
                 kernel_sizes=[8,5,3], nb_filters=64, depth=3):
        
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.kernel_sizes = kernel_sizes
        self.nb_filters = nb_filters
        self.depth = depth
        
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
        return
    
    def _residual_block(self, input_tensor, current_depth):
        filter_multiplier = [1, 2, 2, 4] # maximum depth is currently 4 blocks!
        n_feature_maps = self.nb_filters*filter_multiplier[current_depth]
        
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=self.kernel_sizes[0], padding='same')(input_tensor)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = keras.layers.Dropout(0.2)(conv_x) # ADDED DROPOUT 

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=self.kernel_sizes[1], padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = keras.layers.Dropout(0.2)(conv_y) # ADDED DROPOUT 

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=self.kernel_sizes[2], padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum if number of channels are inequal
        if input_tensor.shape[-1]!=n_feature_maps:
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_tensor)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        # no need to expand channels if they are equal
        else:
            shortcut_y = keras.layers.BatchNormalization()(input_tensor)

        output_block = keras.layers.add([shortcut_y, conv_z])
        output_block = keras.layers.Activation('relu')(output_block)
        output_block = keras.layers.Dropout(0.2)(output_block) # ADDED DROPOUT

        return output_block         
        

    def build_model(self, input_shape, nb_classes):

        input_layer = keras.layers.Input(input_shape)
        
        x = input_layer
        for d in range(self.depth):
            x = self._residual_block(x, d)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.lr), # standard learning_rate = 0.001
                      metrics=['accuracy'])


        return model
    

    # fit added by Martijn de Jong
    def fit(self, x_train, y_train, x_test, y_test):
        # Callback function to save checkpoints
        checkpoint_path = self.output_directory+'best_model.hdf5'                                                      
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=False,
            save_best_only=True,
            monitor = 'val_loss')
        # Callback function to reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=self.min_lr) #min_lr=0.0001
        
        start_time = time.time()
        # fit model with specified batch size for provided number of epochs
        hist = self.model.fit(x_train, 
                 y_train,
                 batch_size = self.batch_size,
                 epochs = self.nb_epochs,
                 validation_data = (x_test, y_test),
                 shuffle = True,
                 callbacks=[cp_callback, reduce_lr])
        
        # log how long model trained
        self.duration = time.time()-start_time
        
        # save history to dataframe and save dataframe as .csv
        hist_df = pd.DataFrame(hist.history)
        hist_df.index+=1
        hist_df.to_csv(self.output_directory + 'history.csv')
        
        index_best_model = hist_df['val_loss'].idxmin()
        row_best_model = hist_df.loc[index_best_model]
        self.best_acc = row_best_model['val_accuracy']
        self.best_loss = row_best_model['val_loss']
        self.best_lr = row_best_model['lr']
        self.best_ep = index_best_model
        
        self.predict(x_test, y_test, save_metrics=True)
        
        keras.backend.clear_session()              
        return hist_df, self.best_acc, self.best_loss, self.best_ep
    
    # predict added by Martijn de Jong
    def predict(self, x, y, save_metrics=False):
        # load in trained network
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        
        y_pred = model.predict(x, batch_size=self.batch_size)
        
        if save_metrics == False:
            return y_pred
        
        else:
            # calculate metrics of trained network
            y_pred_int = np.argmax(y_pred, axis=1)
            y_test_int = np.argmax(y, axis=1)
            
            equal = np.equal(y_pred_int, y_test_int)
            
            equal_N = equal[np.where(y_pred_int==0)]
            TN = sum(equal_N)
            FN = len(equal_N)-TN
            
            equal_P = equal[np.where(y_pred_int==1)]
            TP = sum(equal_P)
            FP = len(equal_P)-TP
            
            P_precision = TP/(TP+FP)
            P_recall = TP/(TP+FN)
            P_F1 = 2*P_precision*P_recall/(P_precision+P_recall)
            P_bias = (TP+FP)/(TP+FP+TN+FN)
            
            N_precision = TN/(TN+FN)
            N_recall = TN/(TN+FP)
            N_F1 = 2*N_precision*N_recall/(N_precision+N_recall)
            N_bias = (TN+FN)/(TP+FP+TN+FN)
            
            accuracy = (TP+TN)/(TP+FP+TN+FN)
            
            # put metrics in dictionary, convert to dataframe, and save as .csv
            metrics = {'best_ep':self.best_ep, 
                'val_loss':self.best_loss, 'val_acc':self.best_acc, 'fin_lr':self.best_lr,
                'duration':self.duration, 'TP':TP, 'FP': FP, 'TN':TN, 'FN':FN, 'accuracy': accuracy,
                'P_precision': P_precision, 'P_recall': P_recall, 'P_F1': P_F1, 'P_bias':P_bias,
                'N_precision': N_precision, 'N_recall': N_recall, 'N_F1': N_F1, 'N_bias':N_bias}
            
            metric_df = pd.DataFrame(metrics, index=[0])
            metric_df.to_csv(self.output_directory + 'metrics.csv', index=False)