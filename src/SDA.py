from __future__ import print_function
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import GaussianNoise
import seaborn as sn
from sklearn.svm import SVC
import pickle
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from pickle import load
from keras.utils.vis_utils import plot_model
import os
from keras import backend as K
import psutil
import configparser
import paramiko
import socket
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/config', 'config.ini'))
class Sda:

    def __init__( self, n_layers, num_epoch, en_act_func, dec_act_func, loss_func, optimizer, num_classes, n_hid, dropout, batch_size, mode, hostname,rp):#,sftp
        self.n_layers = n_layers
        self.num_epoch = num_epoch
        self.en_act_func = en_act_func
        self.dec_act_func = dec_act_func
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.n_hid = n_hid
        self.dropout = dropout
        self.batch_size = batch_size
        self.mode = mode
        self.hostname = hostname
        self.rp = rp

    def get_sda_dnn(self, x_train, y_train, x_test, data, protocol):


        encoders = []
        x_val_next_layer = []
        x_train_next_layer = []
        np.set_printoptions(precision=8)


        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)

        for cur_layer in range(self.n_layers):

            if cur_layer == 0:
                x_train_cur_layer = x_train
                x_val_cur_layer = x_val
                x_test_cur_layer = x_test
            else:
                x_train_cur_layer = x_train_next_layer
                x_val_cur_layer = x_val_next_layer
                x_test_cur_layer = x_test_next_layer

            print('Input for layer ' + str(cur_layer), ": " , x_train_cur_layer)

            print(np.count_nonzero(x_train_cur_layer))

            print("Training layer " + str(cur_layer))
            # create model
            cur_model = KerasRegressor(build_fn=self.create_encoder_model_cv, cur_layer= cur_layer,  x_train_cur_layer_shape=x_train_cur_layer.shape[1], epochs=self.num_epoch, batch_size = self.batch_size, verbose=2)
            hidden_layers = [int(i) for i in config['SDAE']['n_hid'].split(',')]
            dropout = [float(i) for i in config['SDAE']['dropout'].split(',')]
            batch_size = [int(i) for i in (config['SDAE']['batch_size'])]
            param_grid = dict(hidden_layers = hidden_layers, dropout = dropout, batch_size=batch_size)
            grid = GridSearchCV(estimator=cur_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2, n_jobs=-1, verbose=5)
            grid.fit(x_train_cur_layer, x_train_cur_layer)

            
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            params = grid.cv_results_['params']
            best_score = -1000
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
                if mean > best_score:
                    best_score = mean
                    best_param = param
            
            self.n_hid[cur_layer] = best_param['hidden_layers']
            self.dropout[cur_layer] = best_param['dropout']
            self.batch_size = best_param['batch_size']
            cur_model = self.create_encoder_model_best_params(cur_layer, x_train_cur_layer.shape[1])
            

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

            h = cur_model.fit_generator(generator= data.batch_generator_custom(
                x_train_cur_layer, x_train_cur_layer,
                self.batch_size,
                True),
                callbacks=[early_stopping],
                nb_epoch=self.num_epoch,
                steps_per_epoch= x_train_cur_layer.shape[0] // self.batch_size,
                verbose=2,
                validation_data= data.batch_generator_custom(
                    x_val_cur_layer, x_val_cur_layer,
                    self.batch_size,
                    False),
                nb_val_samples=x_val_cur_layer.shape[0]
            )

            print("Layer " + str(cur_layer) + " has been trained")


            encoder_layer = cur_model.layers[-2]

            encoders.append(encoder_layer)


            x_train_next_layer = self.get_nth_layer_encoder_output(cur_model, x_train_cur_layer, self.n_hid[cur_layer], data, 2)
            print(np.count_nonzero(x_train_next_layer))

            x_val_next_layer = self.get_nth_layer_encoder_output(cur_model, x_val_cur_layer, self.n_hid[cur_layer], data, 2)

            x_test_next_layer = self.get_nth_layer_encoder_output(cur_model, x_test_cur_layer, self.n_hid[cur_layer], data, 2)
        

        x_train_final_layer = x_train_next_layer
        
        x_val_final_layer = x_val_next_layer
        x_test_final_layer = x_test_next_layer
        
        if self.mode == 'Retraining':
            
            if self.hostname ==  config['server1']['hostname']:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())          
                ssh.connect(config['server2']['ip_address'],
                username = config['server2']['username'], password = config['server2']['password'], allow_agent = False)
                sftp = ssh.open_sftp()
                self.sftp.chdir(os.path.join(self.rp,'models'))
                self.sftp.get('{}_feature_extractor.h5'.format(protocol),os.path.join(self.rp,'models','{}_feature_extractor.h5'.format(protocol)) )
                sdae_model = tf.keras.models.load_model(os.path.join(self.rp, 'models',protocol) + "_feature_extractor.h5")
            else:
                sdae_model = tf.keras.models.load_model(os.path.join(self.rp,'models', protocol) + "_feature_extractor.h5")

        if self.mode == 'TrainingFromScratch':
            sdae_model = Sequential()
            for en in range(len(encoders)):
                sdae_model.add(encoders[en])
        sdae_model.save(os.path.join(self.rp ,'models',protocol)+ "_feature_extractor.h5")
        if self.hostname == config['server1']['hostname']:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(config['server1']['ip_address'],
                username = config['server1']['username'], password = config['server1']['password'], allow_agent = False)
                sftp = ssh.open_sftp()
                sftp.chdir(os.path.join(self.rp,'models'))
                sftp.get('{}_feature_extractor.h5'.format(protocol),os.path.join(self.rp,'models','{}_feature_extractor.h5'.format(protocol))) 
        return sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer


    def create_encoder_model_cv( self, cur_layer=0, x_train_cur_layer_shape=0, hidden_layers = 1, dropout = 0.01):
        input_layer = Input(shape=(x_train_cur_layer_shape,))

        dropout_layer = Dropout(dropout)

        in_dropout = dropout_layer(input_layer)

        encoder_layer = Dense(input_shape=(x_train_cur_layer_shape,), output_dim= hidden_layers,
                              init='he_uniform',
                              activation=self.en_act_func[cur_layer], name='encoder' + str(cur_layer)
                              )
        encoder = encoder_layer(in_dropout)

        n_out = x_train_cur_layer_shape # same no. of output units as input units (to reconstruct the signal)

        

        decoder_layer = Dense(output_dim=n_out, init='he_uniform',
                              activation=self.dec_act_func[cur_layer], name='decoder' + str(cur_layer))
        decoder = decoder_layer(encoder)

        cur_model = Model(input_layer, decoder)

        cur_model.compile(loss='mse', optimizer=self.optimizer, metrics= ['mse'])

        cur_model.summary()
        return  cur_model

    def create_encoder_model_best_params( self, cur_layer, x_train_cur_layer_shape):
        input_layer = Input(shape=(x_train_cur_layer_shape,))

        dropout_layer = Dropout(self.dropout[cur_layer])

        in_dropout = dropout_layer(input_layer)

        encoder_layer = Dense(input_shape=(x_train_cur_layer_shape,), output_dim=self.n_hid[cur_layer],
                              init='he_uniform',
                              activation=self.en_act_func[cur_layer], name='encoder' + str(cur_layer)
                              )
        encoder = encoder_layer(in_dropout)

        n_out = x_train_cur_layer_shape  # same no. of output units as input units (to reconstruct the signal)

        

        decoder_layer = Dense(output_dim=n_out, init='he_uniform',
                              activation=self.dec_act_func[cur_layer], name='decoder' + str(cur_layer))
        decoder = decoder_layer(encoder)

        cur_model = Model(input_layer, decoder)

        cur_model.compile(loss='mse', optimizer=self.optimizer, metrics=['mse'])

        cur_model.summary()
        return cur_model

    def get_trained_classifier( self, model, x_train, y_train, data, protocol, training):

            #build a new model
            if(training == 'trainFromScratch'):
                sdae_model = Sequential()

                sdae_model.add(model)

                sdae_model.add(Dense(self.num_classes, activation='softmax'))

                sdae_model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

                sdae_model.summary()

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
            elif (training == 'continueTraining'):
                
                if self.hostname ==  config['server1']['hostname']:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        
                    ssh.connect(config['server2']['ip_address'],
                        username = config['server2']['username'], password = config['server2']['password'], allow_agent = False)
                    sftp = ssh.open_sftp()
                    sftp.chdir(os.path.join(self.rp,'models'))
                    sftp.get('{}_SDAE_model.h5'.format(protocol),os.path.join(self.rp,'models','{}_SDAE_model.h5'.format(protocol)))
                    sdae_model = load_model(os.path.join(self.rp,"models",protocol) + "_" + "SDAE" + "_model.h5")
                else:
                    sdae_model = load_model("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/" + protocol + "_" + "SDAE" + "_model.h5")
                early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)


            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train)
            
            h = sdae_model.fit_generator(generator=data.batch_generator_custom(x_train, to_categorical(y_train),
                                                                          self.batch_size,
                                                                          True
                                                                          ),
                                       callbacks=[early_stopping],
                                       nb_epoch=self.num_epoch,
                                       steps_per_epoch=x_train.shape[0] // self.batch_size,
                                       verbose=2,
                                       validation_data=data.batch_generator_custom(x_val, to_categorical(y_val),
                                                                                self.batch_size,
                                                                                False
                                                                                ),
                                           nb_val_samples=x_train.shape[0])
            return sdae_model

    def get_DNN( self, x_train, y_train, data, protocol):

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train)

        model = Sequential()
        model.add(Dense(75, input_dim=x_train.shape[1] , activation = 'relu'))
        model.add(Dense(50, activation = 'relu'))
        model.add(Dense(20, activation= 'relu'))
        model.add(Dense(self.num_classes,activation = 'softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        h = model.fit_generator(generator=data.batch_generator_custom(x_train, to_categorical(y_train),
                                                                           self.batch_size,
                                                                           True
                                                                           ),
                                     callbacks=[early_stopping],
                                     nb_epoch=self.num_epoch,
                                     steps_per_epoch=x_train.shape[0] // self.batch_size,
                                     verbose=2,
                                     validation_data=data.batch_generator_custom(x_val, to_categorical(y_val),
                                                                                 self.batch_size,
                                                                                 False
                                                                                 ),
                                     nb_val_samples=x_train.shape[0])

        return model

    def get_nth_layer_encoder_output(self, model, input, output_size, data, nth_layer):

        model_output = np.zeros(shape = (input.shape[0], output_size))
        
        index = 0
        for batch_x, batch_y in data.batch_generator_custom(input, input, 200, False):
                while index <= int(np.ceil(input.shape[0]/200)):
                    model_encoder_output_batch_x = K.function([model.layers[0].input, K.learning_phase()],
                                                              [model.layers[nth_layer].output])
                    model_output[index:index+len(batch_x), :] = model_encoder_output_batch_x([batch_x, 0])[0]
                    index += 1
                break
        return model_output
