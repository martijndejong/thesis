# -*- coding: utf-8 -*-
"""
Original code by Ismail Fawaz

Additions by Martijn de Jong
"""

import keras
import numpy as np
import pandas as pd
import time


class Classifier_INCEPTION():

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=30,
                 lr=0.001, min_lr=0.0001):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.min_lr = min_lr

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            # self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x) # ADDED DROPOUT 
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.2)(x) # ADDED DROPOUT 
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.lr), 
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