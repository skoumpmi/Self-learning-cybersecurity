import os
from os import  path
import numpy as np
from keras.models import load_model
from SDA import Sda
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import seaborn as sn
from sklearn.svm import SVC
import pickle
from pickle import load
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import configparser
import paramiko
import socket
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/config', 'config.ini'))



class Training:

    def __init__( self, x_train, y_train, x_test, y_test, data, protocol, model_type, cv, num_classes, attackLabelsToInt, mode, hostname, rp,target_names):#target_names,
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data = data
        self.protocol = protocol
        self.classifier_type = model_type
        self.cv = cv
        self.num_classes = num_classes
        self. attackLabelsToInt =  attackLabelsToInt
        self.mode = mode
        self.hostname = hostname
        self.rp = rp
        self.target_names = target_names
        

    def classify(self):
       

        
        if(self.classifier_type == 'feature_extractor'):

            print('Training Random Forest model with Stacked Denoising Autoeconder as feature extractor')
            pretrained_sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer = Sda.get_sda_dnn(
                    self.x_train, self.y_train, self.x_test, self.data, self.protocol)


            train = pretrained_sdae_model.predict_generator(generator=self.data.test_generator(self.x_train,
                                                                                          32,
                                                                                          ),
                                                            steps=np.ceil(self.x_train.shape[0] / 32)
                                                            )

            test = pretrained_sdae_model.predict_generator(generator=self.data.test_generator(self.x_test,
                                                                                         32,
                                                                                         ),
                                                           steps=np.ceil(self.x_test.shape[0] / 32)
                                                           )

            model = RandomForestClassifier(n_estimators=200, random_state=0)
            print(train.shape, self.x_train.shape)
            model.fit(train, self.y_train)
            # save the classifier
            with open(os.path.join(self.rp, 'models', self.protocol) + 'SDAE_Random_Forest_model.pkl', 'wb') as fid:
                pickle.dump(model, fid)
    
 
        if (self.classifier_type == 'KNN'):
                print('Training KNN model')
                n_neighbors = [int (i) for i in config['KNN']['n_neighbors'].split(',')]
                leaf_size = [int (i) for i in config['KNN']['leaf_size'].split(',')]
                if len(n_neighbors) == 1:
                    n_neighbors = [int(config['KNN']['n_neighbors'])]
                if len(leaf_size) == 1:
                    leaf_size = [int(config['KNN']['leaf_size'])]
                grid = GridSearchCV(KNeighborsClassifier(), param_grid=dict(n_neighbors = n_neighbors, leaf_size = leaf_size), scoring='f1_macro', cv=self.cv, n_jobs=-1, verbose=int(config['KNN']['verbose']))
                grid.fit(self.x_train, self.y_train)
                best_score = 0
                for mean, stdev, param in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score'], grid.cv_results_['params']):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: n_neighbors=%d and leaf_size=%d ' % (best_score, best_param['n_neighbors'], best_param['leaf_size']))
                model = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], leaf_size=best_param['leaf_size'])
                model.fit(self.x_train, self.y_train)
                prediction = model.predict(self.x_test)
                
        if (self.classifier_type == 'SDAE'):
                    num_classes = self.num_classes
                    print('Training SDAE model')
                    
                    sdae = Sda(n_layers=int(config['SDAE']['n_layers']), 
                            num_epoch=int(config['SDAE']['num_epoch']), 
                            en_act_func= config['SDAE']['en_act_func'].split(','),
                            dec_act_func = config['SDAE']['dec_act_func'].split(','),
                            loss_func=config['SDAE']['loss_func'],
                            optimizer=config['SDAE']['optimizer'],
                            num_classes = self.num_classes, 
                            n_hid= [int(i) for i in config['SDAE']['n_hid'].split(',')],
                            dropout=[float(i) for i in config['SDAE']['dropout'].split(',')],
                            batch_size=int(config['SDAE']['batch_size']),
                            mode = self.mode,
                            hostname = self.hostname,
                            rp=self.rp)
                    if self.mode == 'TrainingFromScratch':
                    
                        pretrained_sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer = sdae.get_sda_dnn(
                                self.x_train, self.y_train, self.x_test, self.data, self.protocol)
                        model = sdae.get_trained_classifier(pretrained_sdae_model, self.x_train, self.y_train,
                                                                    self.data, self.protocol, training='trainFromScratch')
                    if self.mode == 'Retraining':
                        
                        if self.hostname ==  config['server1']['hostname']:
                            ssh = paramiko.SSHClient()
                            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            
                            ssh.connect(config['server2']['ip_address'],
                            username = config['server2']['username'], password = config['server2']['password'], allow_agent = False)
                            sftp = ssh.open_sftp()
                            sftp.chdir(os.path.join(self.rp, 'models'))
                            sftp.get('{}_feature_extractor.h5'.format(self.protocol),os.path.join(self.rp,'models','{}_feature_extractor.h5'.format(self.protocol)) )
                            pretrained_sdae_model = tf.keras.models.load_model(os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),'models' , self.protocol + "_feature_extractor.h5"))
                            sftp.close()
                        else:
                            pretrained_sdae_model = tf.keras.models.load_model(os.path.join(self.rp, 'models',self.protocol) + "_feature_extractor.h5")
                        model = sdae.get_trained_classifier(pretrained_sdae_model, self.x_train, self.y_train,
                                                                self.data, self.protocol, training='continueTraining')
                    prediction = model.predict_generator(generator=self.data.test_generator(self.x_test,
                                                                                                     32,
                                                                                                     ),
                                                                  steps=np.ceil(self.x_test.shape[0] / 32)                                                )
                    # Confusion Matrix and Classification Report
                    prediction = np.argmax(prediction, axis=1)
                    ##cm = confusion_matrix(self.y_test, prediction)
                    print(model.summary())
                    


        if (self.classifier_type == 'Random-Forest'):
                print('Training Random Forest model')
                n_estimators = [int(i) for i in config['Random-Forest']['n_estimators'].split(',')]
                if len(n_estimators) == 1:
                    n_estimators = [int(config['Random-Forest']['n_estimators'])]
                grid = GridSearchCV(RandomForestClassifier(), param_grid=dict(n_estimators=n_estimators), scoring='f1_macro', cv=self.cv, n_jobs=-1,verbose= int(config['Random-Forest']['verbose']))
                grid.fit(self.x_train, self.y_train)
                best_score = 0
                for mean, stdev, param in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score'], grid.cv_results_['params']):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: n_estimators=%d' % (best_score, best_param['n_estimators']))
                model = RandomForestClassifier(n_estimators=best_param['n_estimators'], random_state=0)
                model.fit(self.x_train, self.y_train)
                prediction = model.predict(self.x_test)
                ##cm = confusion_matrix(self.y_test, prediction)

               
        if (self.classifier_type == 'LogReg'):
                print('Traingin Logistic regression model')
                max_iter= [int(i) for i in config['Log_Reg']['max_iter'].split(',')]
                if len(max_iter) == 1:
                    max_iter = [int(config['Log_Reg']['max_iter'])]
                grid = GridSearchCV(LogisticRegression(), param_grid=dict(max_iter=max_iter), scoring=config['Log_Reg']['scoring'], cv=self.cv,
                                    n_jobs=-1, verbose= int(config['Log_Reg']['verbose']))
                grid.fit(self.x_train, self.y_train)
                best_score = 0
                for mean, stdev, param in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score'], grid.cv_results_['params']):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: max_iter=%d' % (best_score, best_param['max_iter']))
                model = LogisticRegression(max_iter=best_param['max_iter'])
                model.fit(self.x_train, self.y_train)
                prediction = model.predict(self.x_test)
        
        target_names = self.target_names
        
        report = classification_report(self.y_test, prediction, target_names=target_names, output_dict=True)
        
        return  model,report
   

