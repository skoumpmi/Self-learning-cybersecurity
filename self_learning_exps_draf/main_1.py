'''
@author vak
'''
#from comet_ml import Experiment
from SDA import Sda
from training import Training
import pandas as pd
from data_process import Data
import sys
import os.path
from os import path
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle
from os import  path
#from pyspark.context import SparkContext
#from pyspark import SparkConf
#from pyspark.sql import SparkSession
from joblib import delayed, Parallel
from functools import partial
from joblibspark import register_spark
import time
from networkFlowsConsumer import SmartHomeKafkaConsumer_SH
import json
import pickle
import pymongo
import os
from pickle import load
#os.environ["PYSPARK_PYTHON"]="/grid/01/spearuser/anaconda3/envs/venv/bin/python3"

#conf = SparkConf().setAppName("Self-Training framework") \
#                  .setMaster("spark://hpc-mgmt.iti.gr:7077")\
#                  .set('spark.executor.cores', '6') \
#                  .set('spark.executor.memory', '20g')
#sc = SparkContext(conf=conf)
#sc.addPyFile("SDA.py")
#sc.addPyFile("data_process.py")

#spark = SparkSession.builder \
#                    .config(conf=conf) \
#                    .getOrCreate()

#sc = SparkContext("local[*]", "Single node self-training")

def main():

    #connect to mongodb database
    client = pymongo.MongoClient('localhost', 27017)
    SPEAR_NetflowsDatabase = client.labelled_netflows_db
    #one collection for each protocol
    labelled_mqtt = SPEAR_NetflowsDatabase["labelled_mqtt"]
    labelled_modbus = SPEAR_NetflowsDatabase["labelled_modbus"]
    labelled_bacnet = SPEAR_NetflowsDatabase["labelled_bacnet"]
    labelled_radius = SPEAR_NetflowsDatabase["labelled_radius"]
    #labelled_ntp = SPEAR_NetflowsDatabase["labelled_ntp"]
    kafkacons = SmartHomeKafkaConsumer_SH()
    print("Created: Kafka Consumer obj\n")
    topic = "flows_topic"
    kafkacons.consumer.subscribe([topic])
    print("Consumer subscribed in Topic: [\"" + topic + "\"]\n")
    for message in kafkacons.consumer:
        str_protocol = message.value['str_protocol']
        print('===============================')
        print('PROTOCOL IS')
        print(str_protocol)
        print('================================')
        if (str_protocol == 'Modbus' or str_protocol == 'Mqtt' or str_protocol == 'Bacnet' or str_protocol == 'Ntp'or str_protocol == 'Radius'):
        #if (str_protocol == 'Modbus'  or str_protocol == 'Bacnet' or str_protocol == 'Ntp'or str_protocol == 'Radius'):
            if str_protocol == 'Modbus':
                protocol = 'MODBUS'
                attackLabelsToInt = {"Label": {"Normal": 0, "UID brute force": 1, "Enumeration Function": 2, "Fuzzing Read Holding Registers": 3}}
            elif str_protocol == 'Mqtt':
                protocol = 'MQTT'
                attackLabelsToInt = {"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
            elif str_protocol == 'Bacnet':
                protocol = 'BACNET'
                attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
            #elif str_protocol == 'Ntp':
                #protocol = 'NTP'
                #attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
            elif str_protocol == 'Radius':
                protocol = 'RADIUS'
                attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
            print(" --- New " + str_protocol + " Flow --- : " + str(message.value) + "\n")
            # load_best_model
            fileName = os.listdir("best_models/" + protocol)[0]
            modelType = fileName.split(sep="_")
            if modelType[1] == 'SDAE':
                model = load_model("best_models/" + protocol + "/" + protocol + "_" + modelType[1] + "_model.h5")
            else:
                model = pickle.load(open('best_models/' + protocol + "/" + protocol + "_" + modelType[1] + '_model.pkl', 'rb'))
            #load the scaler
            scaler = load(open('scalers/' + protocol + "_" + modelType[1] + '_scaler.pkl', 'rb'))
            netflow_df = pd.DataFrame(list(message.value.items())).T
            # grab the first row for the header
            new_header = netflow_df.iloc[0]
            # take the data less the header row
            netflow_df = netflow_df[1:]
            # set the header row as the df header
            netflow_df.columns = new_header
            netflow_df = netflow_df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Protocol", "Timestamp", "Src Port", "Dst Port"]).iloc[:, 0:76]
            print(netflow_df)
            # drop object dtypes
            for col in netflow_df.columns:
                if (netflow_df[col].dtype == np.object):
                    netflow_df[col] = pd.to_numeric(netflow_df[col], errors='coerce')
            # replace infinite values with nan
            netflow_df = netflow_df.replace([np.inf, -np.inf], np.nan)
            # drop nan values
            netflow_df = netflow_df.dropna()
            print(netflow_df)
            netflow_df.info()
            if netflow_df.empty == True:
                print('Netflows DataFrame is empty')
                continue
            # apply scaler
            scaled_netflow = scaler.transform(netflow_df)
            prediction = model.predict(scaled_netflow)
            print(prediction[0])
            key = get_key(attackLabelsToInt, prediction[0])
            print(key)
            message.value['Label'] = key
            if protocol == 'MODBUS':
                labelled_modbus.insert(message.value)
            elif protocol == 'MQTT':
                labelled_mqtt.insert(message.value)
            elif protocol == 'BACNET':
                labelled_bacnet.insert(message.value)
            elif protocol == 'RADIUS':
                labelled_radius.insert(message.value)
            #elif protocol == 'NTP':
                #labelled_ntp.insert(message.value)

    print("Kafka Consumer will close.")
    kafkacons.consumer.close()


# function to return key for any value
def get_key(dict, val):
    for key, value in dict['Label'].items():
        if val == value:
            return key
    return "key doesn't exist"

if __name__  == '__main__':
    main()
