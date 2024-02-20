from os import lchown
from flask import Flask, request, jsonify,json, render_template
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
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process
#import retrain_models
#tf.compat.v1.disable_eager_execution()

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# netflow_id = request_body['value']


db = myclient["labelled_netflows_db"]
modbus = db["labelled_modbus"]
mqtt = db["labelled_mqtt"]

mqtt_df = pd.DataFrame(list(mqtt.find())).iloc[:, 1:]
print(mqtt_df['Label'])
mqtt_labels = mqtt_df['Label'].unique()
print(mqtt_labels)
mqtt.update_many({"Label": "normal"},
                   {
                       "$set": {"Label": "Normal"},
                       # $currentDate: { lastModified: true }
                   })
mqtt_df = pd.DataFrame(list(mqtt.find())).iloc[:, 1:]
print(mqtt_df['Label'])

#print(mqtt_labels)
mqtt_labels = mqtt_df['Label'].unique()
print(mqtt_labels)