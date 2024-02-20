from sklearn.model_selection import train_test_split
import numpy as np
#from imblearn.over_sampling import SMOTE, BorderlineSMOTE,ADASYN
from sklearn.utils import shuffle
#from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle
from os import  path
from pickle import load

class Data:

    def __init__( self, data, protocol, model_type):
        self.data = data
        self.protocol = protocol
        self.model_type = model_type

    def batch_generator_custom(self, X, Y,  batch_size, shuffle_data):
        if X.shape == Y.shape:
            Y = None
        if shuffle_data == True:
            if Y is not None:
                X,Y = shuffle(X,Y)
            else:
                X = shuffle(X)
        #print(X[:10])
        #print(Y[:10])
        while True:

                for batch in range(0, X.shape[0], batch_size):

                    if (X.shape[0]-batch) >= batch_size:
                       # try:
                            #print('Length of batch:', batch_size)
                            batch_X = X[batch: batch + batch_size, :]
                            if Y is not None:
                                batch_Y = Y[batch: batch + batch_size]

                                #print(batch, batch_X.shape)
                                yield batch_X, batch_Y
                            else:
                                yield batch_X, np.zeros(batch_X.shape)
                    else:
                            batch_X = X[batch:, :]
                            if Y is not None:
                                batch_Y = Y[batch:]
                                yield batch_X, batch_Y
                            else:
                                yield batch_X, np.zeros(batch_X.shape)

    def test_generator( self, X , batch_size):
        while True:
            try:
                for batch in range(0, X.shape[0], batch_size):

                    if (X.shape[0]-batch) >= batch_size:

                            batch_X = X[batch: batch + batch_size, :]

                            yield batch_X, None

                    else:

                            batch_X = X[batch:, :]
                            yield batch_X, None
            except(StopIteration):
                break

    def preprocess_data( self, x):

        x = x.drop(columns=["Flow ID", "Src IP", "Dst IP", "Protocol", "Timestamp", "Src Port", "Dst Port"])

        # summarize class distribution
        #counter = Counter(data)
        #print(counter)

        X = x.values.astype(np.float)
        # Y = Y.as_matrix().astype(np.float)
        #if not path.exists('scalers/' + self.protocol + "_" + self.model_type +'_scaler.pkl'):
        #   print('scaler for ', self.protocol, 'and', self.model_type, 'does not exist')
        scaler = StandardScaler()
        X = scaler.fit(X).transform(X)
        with open('scalers/' + self.protocol + "_" + self.model_type + '_scaler.pkl', 'wb') as fid:
            pickle.dump(scaler, fid)

        print('scaler for ', self.protocol,' exists')
        #load the scaler
        scaler = load(open('scalers/' + self.protocol + "_" + self.model_type +'_scaler.pkl', 'rb'))
        X = scaler.fit(X).transform(X)
        return X

    def train_test_data(self):
        print(self.data.shape)
        # replace infinite values with nan
        data = self.data.replace([np.inf, -np.inf], np.nan)
        # drop nan values
        data = data.dropna()


        #data.info()
        X = data.iloc[:, 0:83]
        Y = data.iloc[:, -1]
        #X.info()

        #print(Counter(Y))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

        x_train = self.preprocess_data(x_train)
        x_test = self.preprocess_data(x_test)

        if self.protocol == 'MQTT':
            print('MQTT dataset')
            #MQTT dataset
            #oversample = ADASYN(sampling_strategy={2: counter[2]*100, 3: counter[3]*100})
            #X, Y = oversample.fit_resample(X, Y)

        elif self.protocol == 'MODBUS':
            print('MODBUS dataset')
            #MODBUS dataset
            #oversample = ADASYN(sampling_strategy={2: counter[2] * 10})
            #X, Y = oversample.fit_resample(X, Y)

            #undersample = RandomUnderSampler(sampling_strategy={1: counter[1] // 10})
            #X, Y = undersample.fit_resample(X, Y)

        elif self.protocol == 'BACNET':
            print('BACnet dataset')
            # BACnet dataset
            #undersample = RandomUnderSampler(sampling_strategy={1: counter[1] // 100})
            #X, Y = undersample.fit_resample(X, Y)

        elif self.protocol == 'NTP':
            print('NTP dataset')
            #oversample = ADASYN(sampling_strategy={1: counter[1] * 10, 2: counter[2] * 10})
            #X, Y = oversample.fit_resample(X, Y)
            #undersample = RandomUnderSampler(sampling_strategy={0: counter[0] // 100})
            #X, Y = undersample.fit_resample(X, Y)
        #elif self.protocol == 'RADIUS':



        #class distribution for y_train
        counter_ytrain = Counter(y_train)
        print(counter_ytrain)

        # class distribution for y_test
        counter_ytest = Counter(y_test)
        print(counter_ytest)

        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        return x_train, x_test, y_train, y_test
