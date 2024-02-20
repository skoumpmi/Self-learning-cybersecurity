from os import lchown
from flask import Flask, request, jsonify, json, render_template
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
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process

# import retrain_models
# tf.compat.v1.disable_eager_execution()




def post_request():
    
	myclient = pymongo.MongoClient("mongodb://localhost:27017/")
	x1 = pd.DataFrame()
	x2 = pd.DataFrame()
	x3 = pd.DataFrame()

	db = myclient["labelled_netflows_db"]
	modbus = db["labelled_modbus"]
	x1 = pd.DataFrame(list(modbus.find()))
	x2 = pd.DataFrame(list(modbus.find()))
	x3 = pd.DataFrame(list(modbus.find()))
	x_modbus = pd.concat([x1,x2,x3]).reset_index(drop=True)
	csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
	columns = list(set(x_modbus.columns).intersection(csv.columns))
	csv = csv[columns].iloc[-13333:].reset_index(drop=True)
	print(csv)
	x_modbus.to_csv('labelled_modbus.csv')

post_request()
		
       
