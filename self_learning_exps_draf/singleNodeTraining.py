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
from os import  path
from keras.wrappers.scikit_learn import KerasRegressor
import spark_sklearn
from sklearn.model_selection import GridSearchCV
#from spark_sklearn.grid_search import GridSearchCV
#from skspark.model_selection import GridSearchCV
from pyspark.context import SparkContext
from pyspark import SparkConf
#from sklearn import grid_search
from pickle import load
from keras.utils.vis_utils import plot_model
from pyspark.sql import SparkSession
from spark_sklearn.util import createLocalSparkSession
#sc = createLocalSparkSession().sparkContext
#sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
#sc = SparkContext("local", "first app")
import os


class Sda:

    def __init__( self, n_layers, num_epoch, batch_size, num_hid, dropout, en_act_func, dec_act_func, loss_func, optimizer, num_classes):
        self.n_layers = n_layers
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.n_hid = num_hid
        self.dropout= dropout
        self.en_act_func = en_act_func
        self.dec_act_func = dec_act_func
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.num_classes = num_classes





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

            hidden_layers = [200,150,100,50,25]
            dropout = [0.01, 0.05, 0.10]
            batch_size = [32, 64, 128]
            param_grid = dict(hidden_layers = hidden_layers, dropout = dropout, batch_size = batch_size)
            grid = GridSearchCV(estimator=cur_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=2)
            grid.fit(x_train_cur_layer, x_train_cur_layer)

            # summarize results
            #print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            params = grid.cv_results_['params']
            best_score = -1000
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
                if mean > best_score:
                    best_score = mean
                    best_param = param
            print('Best score: %f with best params: %d %f %d'% (best_score, best_param['hidden_layers'], best_param['dropout'], best_param['batch_size']))
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
        print('All layers have been trained')

        x_train_final_layer = x_train_next_layer
        print("x train features:")
        x_val_final_layer = x_val_next_layer
        x_test_final_layer = x_test_next_layer
        print('x_test features', x_test_final_layer)

        sdae_model = Sequential()
        #input_layer = Input(shape=(encoders[0].input_shape[1],))
        #dropout_layer = Dropout(self.dropout[0])(input_layer)
        #previous_layer = dropout_layer


        for en in range(len(encoders)):
            #en.inbound_nodes = []
            #encoder = encoders[en](previous_layer)
            #previous_layer = encoder
            sdae_model.add(encoders[en])

        #sdae_model = Model(input_layer, encoder)
            #sdae_model.add(encoders[en])
            #if  (en != len(encoders)-1):
                #sdae_model.add(BatchNormalization())
        print(sdae_model.summary())
        sdae_model.save('models/' + protocol + "_feature_extractor.h5")
        return sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer


    def create_encoder_model_cv( self, cur_layer=0, x_train_cur_layer_shape=0, hidden_layers = 1, dropout = 0.01, batch_size = 32):
        input_layer = Input(shape=(x_train_cur_layer_shape,))

        # noiz = GaussianNoise(0.01, input_shape=(x_train_cur_layer.shape[1],))

        # input_layer = noiz(input_layer)
        dropout_layer = Dropout(self.dropout[cur_layer])

        in_dropout = dropout_layer(input_layer)

        encoder_layer = Dense(input_shape=(x_train_cur_layer_shape,), output_dim= hidden_layers,
                              init='he_uniform',
                              activation=self.en_act_func[cur_layer], name='encoder' + str(cur_layer)
                              )
        encoder = encoder_layer(in_dropout)

        # enc_dropout = dropout_layer(encoder)

        # enc_norm = BatchNormalization()(enc_dropout)

        n_out = x_train_cur_layer_shape # same no. of output units as input units (to reconstruct the signal)

        print('Size of output for layer' + str(cur_layer), ": ", n_out)

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

        # enc_dropout = dropout_layer(encoder)
        # enc_norm = BatchNormalization()(enc_dropout)

        n_out = x_train_cur_layer_shape  # same no. of output units as input units (to reconstruct the signal)

        print('Size of output for layer' + str(cur_layer), ": ", n_out)

        decoder_layer = Dense(output_dim=n_out, init='he_uniform',
                              activation=self.dec_act_func[cur_layer], name='decoder' + str(cur_layer))
        decoder = decoder_layer(encoder)

        cur_model = Model(input_layer, decoder)

        cur_model.compile(loss='mse', optimizer=self.optimizer, metrics=['mse'])

        cur_model.summary()
        return cur_model

    def get_trained_classifier( self, model, x_train, y_train, data, protocol):

            sdae_model = Sequential()

            sdae_model.add(model)

            sdae_model.add(Dense(self.num_classes, activation='softmax'))


            sdae_model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

            sdae_model.summary()

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

            #print(x_train.shape, y_train.shape)

        #kfold = StratifiedKFold(n_splits=5)

        #for train, val in kfold.split(x_train, y_train):
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train)
            print('Training set: ', x_train.shape, y_train.shape)
            print('Validation set: ', x_train.shape, y_train.shape)

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
            # summarize history for loss
            plt.plot(h.history['loss'])
            plt.plot(h.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('reports/' + protocol + '_loss_history')
            plt.show()

            # Plot training & validation accuracy values
            plt.plot(h.history['accuracy'])
            plt.plot(h.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig('reports/' + protocol + '_accuracy_history')
            plt.show()
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
        # summarize history for loss
        plt.plot(h.history['loss'])
        plt.plot(h.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('reports/' + protocol + '_DNN_loss_history')
        plt.show()

        # Plot training & validation accuracy values
        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('reports/' + protocol + '_DNN_accuracy_history')
        plt.show()

        return model

    def test( self, protocol, model_type):
        if protocol == 'MQTT':
            print('MQTT network flow classification test')
            csv = pd.read_csv("testing/MQTT_test.csv")

            attackLabelsToInt = {
                "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
            target_names = ['Unauthorized Subscribe', 'Large Payload', 'Connection Overflow']


        elif protocol == 'MODBUS':
            print('MODBUS network flow classification test')

        elif protocol == 'BACNET':
            print('BACNET network flow classification test')
            csv = pd.read_csv("testing/BACnet_test.csv")
            attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
            target_names = ['BACnet Fuzzing', 'Tampering', 'Flooding']

        elif protocol == 'NTP':
            print('NTP network flow classification test')
            csv = pd.read_csv("testing/NTP_test.csv")
            attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
            target_names = ['KissOfDeath', 'TimeSkimming']

        elif protocol == 'RADIUS':
            print('RADIUS network flow classification test')
            csv = pd.read_csv("testing/Radius_test.csv")
            attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
            target_names = {'Brute Force'}
            
        csv.replace(attackLabelsToInt, inplace=True)
        csv = csv.replace([np.inf, -np.inf], np.nan)
        # drop nan values
        csv = csv.dropna()
        # data.info()
        X = csv.iloc[:, 0:82]
        Y = csv.iloc[:, -1]
        X = X.drop(columns=["Flow ID", "Src IP", "Dst IP", "Protocol", "Timestamp", "Src Port", "Dst Port"])
        # load the scaler
        scaler = load(open('scalers/' + protocol + "_" + model_type +'_scaler.pkl', 'rb'))
        # transform the test dataset
        X_scaled = scaler.transform(X)
        if model_type == 'SDAE':
            model = load_model("models/" + protocol + "_" + model_type+ "_model.h5")
            prediction = model.predict_classes(X_scaled)
        else:
            model = pickle.load(open('models/' + protocol + "_" + model_type +'_model.pkl', 'rb'))
            prediction = model.predict(X_scaled)
        #sdae_model = load_model("models/" + protocol + "_SDAE_model.h5")
        #prediction = sdae_model.predict_classes(network_flow)
        cm = confusion_matrix(Y, prediction)
        pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm)

    def classify( self, x_train, y_train, x_test, y_test, data, protocol, classifier_type):
        #np.set_printoptions(threshold=sys.maxsize)
        #print('Train features:' , x_train[:20])

        #train_features = self.extract_features(model, x_train, data)
        #print('Train features after feature extraction:' , train_features[:100])

        #test_features = self.extract_features(model, x_test, data)
        #print('Test features after feature extraction:' , test_features.shape[:20])
        if(classifier_type == 'feature_extractor'):
            if not path.exists('models/' + protocol + '_feature_extractor.h5'):
                print('Pretrained model does not exist')
                pretrained_sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer = self.get_sda_dnn(
                    x_train, y_train, x_test, data, protocol, sc)
            else:
                print('Loading: ' + 'models/' + protocol + '_feature_extractor.h5')
                pretrained_sdae_model = load_model('models/' + protocol + '_feature_extractor.h5')
                print(pretrained_sdae_model.summary())
            train = pretrained_sdae_model.predict_generator(generator=data.test_generator(x_train,
                                                                                    self.batch_size,
                                                                                    ),
                                                      steps=np.ceil(x_train.shape[0] / self.batch_size)
                                                      )

            test = pretrained_sdae_model.predict_generator(generator=data.test_generator(x_test,
                                                                                    self.batch_size,
                                                                                    ),
                                                      steps=np.ceil(x_test.shape[0] / self.batch_size)
                                                      )

            sdae_rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
            print(train.shape, x_train.shape)
            sdae_rf_model.fit(train, y_train)
            # save the classifier
            with open('models/' + protocol + 'SDAE_Random_Forest_model.pkl', 'wb') as fid:
                pickle.dump(sdae_rf_model, fid)
            prediction = sdae_rf_model.predict(test)
            cm = confusion_matrix(y_test, prediction)

        if(classifier_type == 'SVC'):
            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.pkl'):
                print('SVM model does not exist')
                
                #kernel = ['linear', 'rbf', 'poly']
                C = [0.1, 1, 10]
                kernel = ['linear', 'poly', 'rbf', 'sigmoid']
                param_grid = dict(C = C, kernel = kernel)
                grid = GridSearchCV(SVC(), param_grid=param_grid, scoring = 'f1_macro', cv=2, n_jobs=-1, verbose=5)
                grid.fit(x_train, y_train)
                means = grid.cv_results_['mean_test_score']
                stds = grid.cv_results_['std_test_score']
                params = grid.cv_results_['params']
                best_score = 0
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: kernel=%s and C=%f'% (best_score, best_param['kernel'],best_param['C']))
                SVC_model = SVC(C= best_param['C'], kernel = best_param['kernel'])
                SVC_model.fit(x_train, y_train)
                # save the classifier
                with open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'wb') as fid:
                    pickle.dump(SVC_model, fid)
            else:
                SVC_model = pickle.load(open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'rb'))
            prediction = SVC_model.predict(x_test)
            cm = confusion_matrix(y_test, prediction)


        if(classifier_type == 'KNN'):
            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.pkl'):
                print('KNN model does not exist')
                KNN_model = KNeighborsClassifier(n_neighbors=5)
                KNN_model.fit(x_train, y_train)
                # save the classifier
                with open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'wb') as fid:
                    pickle.dump(KNN_model, fid)
            else:
                KNN_model = pickle.load(open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'rb'))
            prediction = KNN_model.predict(x_test)
            cm = confusion_matrix(y_test, prediction)

        if(classifier_type == 'SDAE'):

            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.h5'):
                print('SDAE model does not exist')
                if not path.exists('models/' + protocol + '_feature_extractor.h5'):
                    print('Pretrained model does not exist')
                    pretrained_sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer = self.get_sda_dnn(x_train, y_train, x_test, data, protocol)
                else:
                    print('Loading: ' + 'models/' + protocol + '_feature_extractor.h5')
                    pretrained_sdae_model = load_model('models/' + protocol + '_feature_extractor.h5')
                    print(pretrained_sdae_model.summary())
                sdae_model = self.get_trained_classifier(pretrained_sdae_model, x_train, y_train, data,
                                                           protocol)
                # save model and architecture to single file
                sdae_model.save('models/' + protocol + "_" + classifier_type+ '_model.h5')
                print("Saved" + protocol + "model to disk")
            else:
                print('Loading: ' + 'models/' + protocol + "_" + classifier_type+ '_model.h5')
                sdae_model = load_model('models/' + protocol + "_" + classifier_type+ '_model.h5')

                print(sdae_model.summary())

            prediction = sdae_model.predict_generator(generator=data.test_generator(x_test,
                                                                                self.batch_size,
                                                                                ),
                                                  steps=np.ceil(x_test.shape[0] / self.batch_size)
                                                  )
            # Confusion Matrix and Classification Report
            prediction = np.argmax(prediction, axis=1)
            cm = confusion_matrix(y_test, prediction)

        if (classifier_type == 'DNN'):
            if not path.exists("models/" + protocol + "_DNN_model.h5"):
                print('DNN model does not exist')
                nn = self.get_DNN(x_train, y_train.to_numpy(), data, protocol)
                # save model and architecture to single file
                nn.save("models/" + protocol + "_DNN_model.h5")
                print("Saved" + protocol + " DNN model to disk")
            else:
                print('Loading: ' + 'models/' + protocol + '_DNN_model.h5')
                dnn_model = load_model("models/" + protocol + "_DNN_model.h5")
                print(dnn_model.summary())

            prediction = dnn_model.predict_generator(generator=data.test_generator(x_test,
                                                                                    self.batch_size,
                                                                                    ),
                                                      steps=np.ceil(x_test.shape[0] / self.batch_size)
                                                      )
            # Confusion Matrix and Classification Report

            prediction = np.argmax(prediction, axis=1)

            cm = confusion_matrix(y_test, prediction)

        if(classifier_type == 'Random-Forest'):
            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.pkl'):
                print('Random Forest model does not exist')
                rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
                rf_model.fit(x_train, y_train)
                # save the classifier
                with open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'wb') as fid:
                    pickle.dump(rf_model, fid)
            else:
                rf_model = pickle.load(open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'rb'))
            prediction = rf_model.predict(x_test)
            cm = confusion_matrix(y_test, prediction)

        if(classifier_type == 'LinearReg'):
            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.pkl'):
                print('Logistic regression model does not exist')
                logreg_model = LogisticRegression(max_iter=500)
                logreg_model.fit(x_train, y_train)
                with open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'wb') as fid:
                    pickle.dump(logreg_model, fid)
            else:
                logreg_model = pickle.load(open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'rb'))
            prediction = logreg_model.predict(x_test)
            cm = confusion_matrix(y_test, prediction)

        if(classifier_type == 'GaussianNB'):
            if not path.exists('models/' + protocol + "_" + classifier_type+ '_model.pkl'):
                print('Gaussian NB model does not exist')
                gnb_model = GaussianNB()
                gnb_model.fit(x_train, y_train)
                with open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'wb') as fid:
                    pickle.dump(gnb_model, fid)
            else:
                gnb_model = pickle.load(open('models/' + protocol + "_" + classifier_type+ '_model.pkl', 'rb'))
            prediction = gnb_model.predict(x_test)
            cm = confusion_matrix(y_test, prediction)

        if protocol == 'MQTT':
            #MQTT dataset
            target_names = ['Normal', 'Unauthorized Subscribe', 'Large Payload', 'Connection Overflow']
        elif protocol == 'MODBUS':
            #modbus dataset
            target_names = ['Normal', 'UID brute force', 'Enumeration Function', 'Fuzzing Read Holding Registers']
        elif protocol == 'BACNET':
            #BACnet dataset
            target_names = ['Normal', 'BACnet Fuzzing', 'Tampering', 'Flooding']
        elif protocol == 'NTP':
            #NTP dataset
            target_names = ['Normal', 'KissOfDeath', 'TimeSkimming']
        elif protocol == 'RADIUS':
            target_names = {'Normal', 'Brute Force'}

        cm = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm)

        cm.to_csv('reports/' + protocol + "_" + classifier_type + "_confusion_matrix.csv")

        report = classification_report(y_test, prediction, target_names=target_names, output_dict = True) 

        report = pd.DataFrame(report).transpose()
        print(report)
        #report.to_csv("reports/" + protocol + "_"  +classifier_type + "_classification_report.csv")

        plt.title('Confusion matrix of_' + classifier_type + '_classifier')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16})
        figure = cm_plot.get_figure()
        figure.savefig('reports/' + protocol + "_" + classifier_type +'_confusion_matrix_heat')

        FP = (cm.sum(axis=0) - np.diag(cm))
        FP_sum = 0
        for fp in FP:
            FP_sum += fp
        print('False Positives: ', FP_sum)

        FN = (cm.sum(axis=1) - np.diag(cm))
        FN_sum = 0
        for fn in FN:
            FN_sum += fn
        print('False Negatives: ', FN_sum)

        TP = np.diag(cm).astype(float)
        TP_sum = 0
        for tp in TP:
            TP_sum += tp
        print('True Positives: ', TP_sum)

        TN = (cm.values.sum() - (FP + FN + TP))
        TN = abs(TN)
        TN_sum = 0
        for tn in TN:
            TN_sum += tn
        print('True Negatives/class: ', TN_sum)
        # Sensitivity, recall, true positive rate

        TPR = TP_sum / (TP_sum + FN_sum)
        print('True positive rate: ', TPR)
        # False positive rate
        FPR = FP_sum / (TN_sum + FP_sum)
        print('False positive rate: ', FPR)
        # Overall Accuracy
        ACC = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum)
        print('Accuracy: ', ACC)


    def get_nth_layer_encoder_output(self, model, input, output_size, data, nth_layer):

        model_output = np.zeros(shape = (input.shape[0], output_size))
        # print('Model output shape:' , model_output.shape)
        # print(model.layers[nth_layer].output)
        index = 0
        for batch_x, batch_y in data.batch_generator_custom(input, input, 200, False):
                while index <= int(np.ceil(input.shape[0]/200)):
                    model_encoder_output_batch_x = K.function([model.layers[0].input, K.learning_phase()],
                                                              [model.layers[nth_layer].output])
                    model_output[index:index+len(batch_x), :] = model_encoder_output_batch_x([batch_x, 0])[0]
                    # print(model_output_batch_x([batch_x, 0])[0])
                    index += 1
                break
        return model_output
