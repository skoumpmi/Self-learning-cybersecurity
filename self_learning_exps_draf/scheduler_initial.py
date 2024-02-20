# Schedule Library imported
import schedule
import time

from os import lchown
from flask import Flask, request, jsonify, json, render_template
import joblibspark
import pymongo
import time
import configparser
import pandas as pd
from multiprocessing import Process
from joblibspark import register_spark
from retraining import Training_ML, Training_DL
from training import Training
from data_process import Data
import pickle
from tensorflow.keras.models import load_model
from pickle import load
from joblib import delayed, Parallel
import time
import numpy as np
#from tensorflow.keras.utils import to_categorical
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
# import tensorflow as tf
import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process
#import psutil
#import findspark
#findspark.init()
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.serializers import PickleSerializer
import os
import sys
from multiprocessing.pool import ThreadPool
import os
import psutil
from sklearn.model_selection import train_test_split
import numpy as np
#from imblearn.over_sampling import SMOTE, BorderlineSMOTE,ADASYN
#from sklearn.utils import shuffle
#from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle
from os import  path
from pickle import load

# gives a single float value


config = configparser.ConfigParser()
config.read('config.ini')
myclient = pymongo.MongoClient('localhost', 27017)
#myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
db = myclient["labelled_netflows_db"]
#modbus = db["labelled_modbus"]



#app = Flask(__name__)
def preprocess_data(x, protocol):

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
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/scalers/' + protocol + "_" + '_scaler.pkl', 'wb') as fid:
            pickle.dump(scaler, fid)

        print('scaler for ', protocol,' exists')
        #load the scaler
        scaler = load(open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/scalers/' + protocol + "_" + '_scaler.pkl', 'rb'))
        X = scaler.fit(X).transform(X)
        return X



#request_body = request.get_json()
    
   

def memory_usage():    
    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    print(modbus.find())
    print(mqtt.find())

    

    x = pd.DataFrame()
    y = pd.DataFrame()

    mqtt_labels = []
    modbus_labels = []
    protocols = ['MQTT', 'MODBUS']

    # the subsets of database records, which correspond to user interaction,
    # are converted to pandas dataframe
    #for item in request_body:
    print('ITEM IS &&&&&&&&&&&&&')
    #print(item)      
    x = pd.concat([x, pd.DataFrame(list(modbus.find()))]).reset_index(drop=True)
    y = pd.concat([y, pd.DataFrame(list(mqtt.find()))]).reset_index(drop=True)
    if x.empty:
        pass
    else:
        protocol = 'MODBUS'
        if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                num_classes = len(list(list(db['modbus_attacks'].find()[0].items())[1][1].values()))
                attackLabelsToInt = {"Label": list(list(db['modbus_attacks'].find())[0].items())[1][1]}
            
        csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/datasets/{}_dataset.csv".format(protocol.lower()))
        # 1. we take into account the historical data from user interactions from the
        # past and now.
        history = db["history_{}".format(protocol.lower())]
        label = db["labelled_{}".format(protocol.lower())]
        label_df = pd.DataFrame(list(label.find())).iloc[:, 1:]
        #label_df = pd.DataFrame(list(label.find())).iloc[:30000, 1:]
        #label_df.to_csv('labelled_modbus_1.csv')
        #CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
        #r'''
        history_df = pd.DataFrame(list(history.find())).iloc[:, 1:]
        all_db_df = pd.concat([history_df, label_df]).reset_index(drop=True)
        columns = list(set(all_db_df.columns).intersection(csv.columns))
        columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
        csv  = csv[columns]
        all_db_df = all_db_df[columns]
        csv = pd.concat([csv , all_db_df]).reset_index(drop=True)
        csv.replace(attackLabelsToInt, inplace=True)
        columns = csv.columns
        #print(columns)
        #print(type(columns))
        for column in columns:
            #print(column)
            if type(csv.iloc[0][column]) == np.float64:
                csv = csv.astype({"{}".format(column): np.float32})
            if type(csv.iloc[0][column]) == np.int64:
                csv = csv.astype({"{}".format(column): np.int32})

        print('======================================')
        ##print('after transformation')
        ##print(csv)
        csv.info(memory_usage="deep")
        csv = csv.replace([np.inf, -np.inf], np.nan)
        # drop nan values
        csv = csv.dropna()
        #data.info()
        X = csv.iloc[:, 0:83]
        Y = csv.iloc[:, -1]
        #X.info()
        #print(Counter(Y))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        x_train = preprocess_data(x_train, protocol)
        x_test = preprocess_data(x_test, protocol)
        #class distribution for y_train
        counter_ytrain = Counter(y_train)
        print(counter_ytrain)

        # class distribution for y_test
        counter_ytest = Counter(y_test)
        print(counter_ytest)
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
        print("CPU Usage = " + CPU_Pct)
        mem=str(os.popen('free -t -m').readlines())
        print(mem) 
        print('******************************************')
        print(psutil.cpu_percent())
        # gives an object with many fields
        print(psutil.virtual_memory())
        # you can convert that object to a dictionary 
        print(dict(psutil.virtual_memory()._asdict()))
        # you can have the percentage of used RAM
        print(psutil.virtual_memory().percent)
        print(psutil.virtual_memory().total)
        print(psutil.virtual_memory().available)
        print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        avlb = psutil.virtual_memory().available
        print(avlb)
        if avlb < 1000000000:
            print(type(avlb))
            db["labelled_{}".format(protocol.lower())].drop()
            # you can calculate percentage of available memory
        
        #'''
    if y.empty:
        pass
    else:
        protocol = 'MQTT'
        if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                num_classes = len(list(list(db['mqtt_attacks'].find()[0].items())[1][1].values()))
                attackLabelsToInt = {"Label": list(list(db['modbus_attacks'].find())[0].items())[1][1]}
            
        csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/datasets/{}_dataset.csv".format(protocol.lower()))
        # 1. we take into account the historical data from user interactions from the
        # past and now.
        history = db["history_{}".format(protocol.lower())]
        label = db["labelled_{}".format(protocol.lower())]
        label_df = pd.DataFrame(list(label.find())).iloc[:, 1:]
        #label_df = pd.DataFrame(list(label.find())).iloc[:150000, 1:]
        #label_df.to_csv('labelled_mqtt_1.csv')
        #CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
        #r'''
        history_df = pd.DataFrame(list(history.find())).iloc[:150000, 1:]
        all_db_df = pd.concat([history_df, label_df]).reset_index(drop=True)
        columns = list(set(all_db_df.columns).intersection(csv.columns))
        columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
        csv  = csv[columns]
        all_db_df = all_db_df[columns]
        csv = pd.concat([csv , all_db_df]).reset_index(drop=True)
        csv.replace(attackLabelsToInt, inplace=True)
        columns = csv.columns
        #print(columns)
        #print(type(columns))
        for column in columns:
            #print(column)
            if type(csv.iloc[0][column]) == np.float64:
                csv = csv.astype({"{}".format(column): np.float32})
            if type(csv.iloc[0][column]) == np.int64:
                csv = csv.astype({"{}".format(column): np.int32})

        print('======================================')
        ##print('after transformation')
        ##print(csv)
        csv.info(memory_usage="deep")
        print(csv.memory_usage(index=True, deep=True))
        csv = csv.replace([np.inf, -np.inf], np.nan)
        # drop nan values
        csv = csv.dropna()
        #data.info()
        X = csv.iloc[:, 0:83]
        Y = csv.iloc[:, -1]
        #X.info()
        #print(Counter(Y))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        x_train = preprocess_data(x_train, protocol)
        x_test = preprocess_data(x_test, protocol)
        #class distribution for y_train
        counter_ytrain = Counter(y_train)
        print(counter_ytrain)
        # class distribution for y_test
        counter_ytest = Counter(y_test)
        print(counter_ytest)
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        CPU_Pct=str(round(float(os.popen('''grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage }' ''').readline()),2))
        print("CPU Usage = " + CPU_Pct)
        mem=str(os.popen('free -t -m').readlines())
        print(mem)
        print('******************************************')
        print(psutil.cpu_percent())
        # gives an object with many fields
        print(psutil.virtual_memory())
        # you can convert that object to a dictionary 
        print(dict(psutil.virtual_memory()._asdict()))
        # you can have the percentage of used RAM
        print(psutil.virtual_memory().percent)
        print(psutil.virtual_memory().total)
        print(psutil.virtual_memory().available)
        # you can calculate percentage of available memory
        print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        avlb = psutil.virtual_memory().available
        print(avlb)
        if avlb < 1000000000:
            print(type(avlb))
            db["labelled_{}".format(protocol.lower())].drop()
            # you can calculate percentage of available memory
        #'''
if __name__ == "__main__":
    not_processing = True
    #print(type(psutil.process_iter()))
    processes = [process.cmdline() for process in psutil.process_iter()]
    print(processes)
    for process in psutil.process_iter():
        #print(process.cmdline())
        if process.cmdline() == ['python', 'self_learning_by_user.py']:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #sys.exit('Process found: exiting.')
            not_processing = False
            #schedule.cancel_job()
    
    print(not_processing)
    if not_processing == True:
            #memory_usage()
            schedule.every(1).minutes.do(memory_usage)
            #schedule.every().hour.do(memory_usage)
            #schedule.every(3).minutes.do(memory_usage)


            # Loop so that the scheduling task
            # keeps on running all time.
            while True :
                processes = [process.cmdline() for process in psutil.process_iter()]
                #print(processes)
                if (['python', 'self_learning_by_user.py']  not in processes):
                    # Checks whether a scheduled task 
                    # is pending to run or not
                    print('NOT IN PROCESSES')
                    schedule.run_pending()
                    #schedule.run_all()
                    time.sleep(1)
                else:
                    schedule.cancel_job(memory_usage)
                    print(psutil.virtual_memory().available)
                    sys.exit()
   
            