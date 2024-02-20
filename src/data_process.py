from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle
from pickle import load
import os


class Data:

    def __init__( self, data, protocol):
        self.data = data
        self.protocol = protocol
        

    def batch_generator_custom(self, X, Y,  batch_size, shuffle_data):
        if X.shape == Y.shape:
            Y = None
        if shuffle_data == True:
            if Y is not None:
                X,Y = shuffle(X,Y)
            else:
                X = shuffle(X)
        while True:

                for batch in range(0, X.shape[0], batch_size):

                    if (X.shape[0]-batch) >= batch_size:
                       
                            batch_X = X[batch: batch + batch_size, :]
                            if Y is not None:
                                batch_Y = Y[batch: batch + batch_size]

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

        X = x.drop(columns=["Flow ID", "Src IP", "Dst IP", "Protocol", "Timestamp", "Src Port", "Dst Port"]).values.astype(np.float)
        
        return X

    def train_test_data(self):
        
        # replace infinite values with nan
        data = self.data.replace([np.inf, -np.inf], np.nan)
        # drop nan values
        data = data.dropna()
        X = data.iloc[:, 0:83]
        Y = data.iloc[:, -1]
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        
        x_train = self.preprocess_data(x_train)
        x_test = self.preprocess_data(x_test)

        if self.protocol == 'MQTT':
            print('MQTT dataset')
            

        elif self.protocol == 'MODBUS':
            print('MODBUS dataset')
            
        print('shapes of training data', x_train.shape,y_train.shape,x_test.shape,y_train.shape)

        #class distribution for y_train
        counter_ytrain = Counter(y_train)
        
        # class distribution for y_test
        counter_ytest = Counter(y_test)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        return x_train, x_test, y_train, y_test
    #THIS FUNCTION IS TRANSFERRED FROM SELF_LEARNING_BY_USER CLASS
    def scaling(self, model_type, rp):
        if not os.path.exists(os.path.join(rp,'scalers',str(self.protocol)) + "_" + str(model_type) +'_scaler.pkl'):
            print('scaler for ', str(self.protocol), 'and', str(model_type), 'does not exist')
            scaler = StandardScaler()
            X = scaler.fit(self.data).transform(self.data)
            with open(os.path.join(rp,'scalers' ,str(self.protocol) + "_" + str(model_type) + '_scaler.pkl'), 'wb') as fid:
                pickle.dump(scaler, fid)

        print('scaler for ', str(self.protocol),' exists')
        #load the scaler
        scaler = load(open(os.path.join(rp,'scalers' ,str(self.protocol)) + "_" + str(model_type) +'_scaler.pkl', 'rb'))
        X = scaler.fit(self.data).transform(self.data)
        return X
