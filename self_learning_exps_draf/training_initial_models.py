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
from SDA import Sda

class Training:

    def __init__( self, x_train, y_train, x_test, y_test, data, protocol, model_type, cv):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.data = data
        self.protocol = protocol
        self.classifier_type = model_type
        self.cv = cv

    def classify(self):
        # np.set_printoptions(threshold=sys.maxsize)
        # print('Train features:' , x_train[:20])

        # train_features = self.extract_features(model, x_train, data)
        # print('Train features after feature extraction:' , train_features[:100])

        # test_features = self.extract_features(model, x_test, data)
        # print('Test features after feature extraction:' , test_features.shape[:20])
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

            sdae_rf_model = RandomForestClassifier(n_estimators=200, random_state=0)
            print(train.shape, self.x_train.shape)
            sdae_rf_model.fit(train, self.y_train)
            # save the classifier
            with open('models/' + self.protocol + 'SDAE_Random_Forest_model.pkl', 'wb') as fid:
                pickle.dump(sdae_rf_model, fid)
            prediction = sdae_rf_model.predict(test)
            cm = confusion_matrix(self.y_test, prediction)

        if(self.classifier_type == 'SVC'):
            print('Training SVM model')
            # kernel = ['linear', 'rbf', 'poly']
            C = [0.1, 1, 10]
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']
            param_grid = dict(C = C, kernel = kernel)
            grid = GridSearchCV(SVC(), param_grid=param_grid, scoring = 'f1_macro', cv=self.cv, n_jobs=-1, verbose=5)
            grid.fit(self.x_train, self.y_train)
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            params = grid.cv_results_['params']
            best_score = 0
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            if mean > best_score:
                best_score = mean
                best_param = param
            print('Best score: %f with best params: kernel=%s and C=%f '% (best_score, best_param['kernel'], best_param['C']))
            SVC_model = SVC(C=best_param['C'], kernel=best_param['kernel'])
            SVC_model.fit(self.x_train, self.y_train)
            # save the classifier
            with open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'wb') as fid:
                pickle.dump(SVC_model, fid)
            SVC_model = pickle.load(open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'rb'))
            prediction = SVC_model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, prediction)

        if (self.classifier_type == 'KNN'):
                print('Training KNN model')
                n_neighbors = [5, 6]
                leaf_size = [1, 2]
                #weights: ['uniform', 'distance']
                #algorithm: ['auto', 'ball_tree', 'kd_tree', 'brute']
                param_grid = dict(n_neighbors = n_neighbors, leaf_size = leaf_size)
                grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, scoring='f1_macro', cv=self.cv, n_jobs=-1, verbose=5)
                grid.fit(self.x_train, self.y_train)
                means = grid.cv_results_['mean_test_score']
                stds = grid.cv_results_['std_test_score']
                params = grid.cv_results_['params']
                best_score = 0
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: n_neighbors=%d and leaf_size=%d ' % (best_score, best_param['n_neighbors'], best_param['leaf_size']))
                KNN_model = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], leaf_size=best_param['leaf_size'])
                KNN_model.fit(self.x_train, self.y_train)
                # save the classifier
                with open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'wb') as fid:
                    pickle.dump(KNN_model, fid)

                KNN_model = pickle.load(open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'rb'))
                prediction = KNN_model.predict(self.x_test)
                cm = confusion_matrix(self.y_test, prediction)

        if (self.classifier_type == 'SDAE'):

                    ''''' try:
                    print('Loading: ' + 'models/' + self.protocol + '_SDAE_model.h5')
                    sdae_model = load_model("models/" + self.protocol + "_SDAE_model.h5")
                    #print(sdae_model.summary())
                    sdae_model = Sda.get_trained_classifier(pretrained_sdae_model, self.x_train, self.y_train,
                                                            self.data, self.protocol, training='continueTraining')

                    # save model and architecture to single file
                    sdae_model.save('models/' + self.protocol + "_" + self.classifier_type + '_model.h5')
                    print("Saved" + self.protocol + "model to disk")

                    prediction = sdae_model.predict_generator(generator=self.data.test_generator(self.x_test,
                                                                                                 32,
                                                                                                 ),
                                                              steps=np.ceil(self.x_test.shape[0] / 32)
                                                              )
                    # Confusion Matrix and Classification Report
                    prediction = np.argmax(prediction, axis=1)
                    cm = confusion_matrix(self.y_test, prediction)
                    f1_new = f1_score(self.y_test, prediction, average='macro')

                    if f1_new < f1_old:'''''
                    if self.protocol == 'MQTT' or self.protocol == 'BACNET' or self.protocol == 'MODBUS':
                        num_classes = 4
                    elif self.protocol == 'NTP':
                        num_classes = 3
                    elif self.protocol == 'RADIUS':
                        num_classes = 2

                    print('Training SDAE model')
                    sdae = Sda(n_layers=2, num_epoch=10, en_act_func=['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
                                   dec_act_func=['linear', 'linear', 'linear', 'linear'], loss_func='mse',
                                   optimizer='adam',
                                   num_classes = num_classes, n_hid=[25,50], dropout=[0.01,0.01], batch_size=32)
                    pretrained_sdae_model, x_train_final_layer, x_val_final_layer, x_test_final_layer = sdae.get_sda_dnn(
                            self.x_train, self.y_train, self.x_test, self.data, self.protocol)
                    print(pretrained_sdae_model.summary())
                    sdae_model = sdae.get_trained_classifier(pretrained_sdae_model, self.x_train, self.y_train,
                                                                self.data, self.protocol, training='trainFromScratch')
                    print(sdae_model.summary())
                    # save model and architecture to single file
                    sdae_model.save('models/' + self.protocol + "_" + self.classifier_type + '_model.h5')
                    print("Saved" + self.protocol + "model to disk")

                    prediction = sdae_model.predict_generator(generator=self.data.test_generator(self.x_test,
                                                                                                     32,
                                                                                                     ),
                                                                  steps=np.ceil(self.x_test.shape[0] / 32)
                                                                  )
                    # Confusion Matrix and Classification Report
                    prediction = np.argmax(prediction, axis=1)
                    cm = confusion_matrix(self.y_test, prediction)

                #except:
                #    print('SDAE model does not exist')


        if (self.classifier_type == 'DNN'):
                print('Training DNN model')
                dnn_model = self.get_DNN(self.x_train, self.y_train.to_numpy(), self.data, self.protocol)
                print(dnn_model.summary())
                # save model and architecture to single file
                dnn_model.save("models/" + self.protocol + "_DNN_model.h5")
                print("Saved" + self.protocol + " DNN model to disk")
                prediction = dnn_model.predict_generator(generator=self.data.test_generator(self.x_test,
                                                                                   32,
                                                                                   ),
                                                     steps=np.ceil(self.x_test.shape[0] / 32)
                                                     )
                # Confusion Matrix and Classification Report
                prediction = np.argmax(prediction, axis=1)
                cm = confusion_matrix(self.y_test, prediction)

        if (self.classifier_type == 'Random-Forest'):
                print('Training Random Forest model')
                #criterion: ['gini', 'entropy']
                n_estimators =  [10, 15]
                #min_samples_leaf: [1, 2, 3]
                #min_samples_split: [3, 4, 5]
                #random_state: [123]
                param_grid = dict(n_estimators=n_estimators)
                grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='f1_macro', cv=self.cv, n_jobs=-1,verbose=5)
                grid.fit(self.x_train, self.y_train)
                means = grid.cv_results_['mean_test_score']
                stds = grid.cv_results_['std_test_score']
                params = grid.cv_results_['params']
                best_score = 0
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: n_estimators=%d' % (best_score, best_param['n_estimators']))
                rf_model = RandomForestClassifier(n_estimators=best_param['n_estimators'], random_state=0)
                rf_model.fit(self.x_train, self.y_train)
                # save the classifier
                with open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'wb') as fid:
                    pickle.dump(rf_model, fid)
                prediction = rf_model.predict(self.x_test)
                cm = confusion_matrix(self.y_test, prediction)

        if (self.classifier_type == 'LogReg'):
                print('Traingin Logistic regression model')
                max_iter=[100, 200, 300]
                param_grid = dict(max_iter=max_iter)
                grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, scoring='f1_macro', cv=self.cv,
                                    n_jobs=-1, verbose=5)
                grid.fit(self.x_train, self.y_train)
                means = grid.cv_results_['mean_test_score']
                stds = grid.cv_results_['std_test_score']
                params = grid.cv_results_['params']
                best_score = 0
                for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
                    if mean > best_score:
                        best_score = mean
                        best_param = param
                print('Best score: %f with best params: max_iter=%d' % (best_score, best_param['max_iter']))
                logreg_model = LogisticRegression(max_iter=best_param['max_iter'])
                logreg_model.fit(self.x_train, self.y_train)
                with open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'wb') as fid:
                    pickle.dump(logreg_model, fid)
                prediction = logreg_model.predict(self.x_test)
                cm = confusion_matrix(self.y_test, prediction)

        if (self.classifier_type == 'GaussianNB'):
                print('Training Gaussian NB model')
                gnb_model = GaussianNB()
                gnb_model.fit(self.x_train, self.y_train)
                with open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'wb') as fid:
                    pickle.dump(gnb_model, fid)
                prediction = gnb_model.predict(self.x_test)
                cm = confusion_matrix(self.y_test, prediction)

        if self.protocol == 'MQTT':
            # MQTT dataset
            target_names = ['Normal', 'Unauthorized Subscribe', 'Large Payload', 'Connection Overflow']
        elif self.protocol == 'MODBUS':
            # modbus dataset
            target_names = ['Normal', 'UID brute force', 'Enumeration Function', 'Fuzzing Read Holding Registers']
        elif self.protocol == 'BACNET':
            # BACnet dataset
            target_names = ['Normal', 'BACnet Fuzzing', 'Tampering', 'Flooding']
        elif self.protocol == 'NTP':
            # NTP dataset
            target_names = ['Normal', 'KissOfDeath', 'TimeSkimming']
        elif self.protocol == 'RADIUS':
            target_names = ['Normal', 'Brute Force']

        cm = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm)

        cm.to_csv('reports/' + self.protocol + "_" + self.classifier_type + "_confusion_matrix.csv")

        report = classification_report(self.y_test, prediction, target_names=target_names, output_dict=True)

        df_report = pd.DataFrame(report).transpose()
        print(df_report)

        # report.to_csv("reports/" + protocol + "_"  +classifier_type + "_classification_report.csv")

        plt.title('Confusion matrix of_' + self.classifier_type + '_classifier')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16})
        figure = cm_plot.get_figure()
        figure.savefig('reports/' + self.protocol + "_" + self.classifier_type + '_confusion_matrix_heat')

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
        return  report

    def test(self):
        if self.protocol == 'MQTT':
            print('MQTT network flow classification test')
            csv = pd.read_csv("testing/MQTT_test.csv")

            attackLabelsToInt = {
                "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
            target_names = ['Unauthorized Subscribe', 'Large Payload', 'Connection Overflow']


        elif self.protocol == 'MODBUS':
            print('MODBUS network flow classification test')

        elif self.protocol == 'BACNET':
            print('BACNET network flow classification test')
            csv = pd.read_csv("testing/BACnet_test.csv")
            attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
            target_names = ['BACnet Fuzzing', 'Tampering', 'Flooding']

        elif self.protocol == 'NTP':
            print('NTP network flow classification test')
            csv = pd.read_csv("testing/NTP_test.csv")
            attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
            target_names = ['KissOfDeath', 'TimeSkimming']

        elif self.protocol == 'RADIUS':
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
        scaler = load(open('scalers/' + self.protocol + "_" + self.classifier_type + '_scaler.pkl', 'rb'))
        # transform the test dataset
        X_scaled = scaler.transform(X)
        if self.classifier_type == 'SDAE':
            model = load_model("models/" + self.protocol + "_" + self.classifier_type + "_model.h5")
            prediction = model.predict_classes(X_scaled)
        else:
            model = pickle.load(open('models/' + self.protocol + "_" + self.classifier_type + '_model.pkl', 'rb'))
            prediction = model.predict(X_scaled)
        # sdae_model = load_model("models/" + protocol + "_SDAE_model.h5")
        # prediction = sdae_model.predict_classes(network_flow)
        cm = confusion_matrix(Y, prediction)
        pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm)



