import sys
from data_creation import data_creation
from data_process import Data
from  training import Training
import paramiko
import socket
from os import lchown
# Get tfrom os import lchown
from flask import Flask, request, jsonify, json, render_template,redirect, url_for
from flask_cors import CORS
import joblibspark
import pymongo
import time
import configparser
import pandas as pd
from multiprocessing import Process
from joblibspark import register_spark
import pickle
from tensorflow.keras.models import load_model
from pickle import load
from joblib import delayed, Parallel
import time
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1 as tf
import psutil
tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process
import requests

import findspark
findspark.init()
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.serializers import PickleSerializer

import os
import sys
from multiprocessing.pool import ThreadPool
import math
import glob
import os
import ssl
import datetime
context=ssl.SSLContext()
context.load_cert_chain('certificates/local/cacert.pem', keyfile='certificates/local/id_rsa.pem')

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/config', 'config.ini'))
myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
db = myclient["labelled_netflows_db"]

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

app = Flask(__name__)
CORS(app)

conf = SparkConf().setAppName('Self_learning_Retrain Framework').setMaster('spark://160.40.53.119:7077').set('spark.executor.memory', '8g').set('spark.worker.cleanup.enabled', True).set('spark.worker.cleanup.interval', 3600).set('spark.worker.cleanup.appDataTtl',86400)#.set('spark.memory.fraction', 0.3).set('spark.memory.storageFraction',0.3)

sc = SparkContext(conf=conf)
print(sc.version)
print(sc._jsc.version())
sc.addPyFile(os.path.join(os.getcwd(), "/src/data_process.py"))
sc.addPyFile(os.path.join(os.getcwd(), "/src/training.py"))
sc.addPyFile(os.path.join(os.getcwd(), "/src/SDA.py"))
sc.addPyFile(os.path.join(os.getcwd(), "/src/data_creation.py"))
sc.addPyFile(os.path.join(os.getcwd(), "/config/config.ini"))
r'''
sc.addPyFile('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/src/data_process.py')
sc.addPyFile('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/src/training.py')
sc.addPyFile('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/src/SDA.py')
sc.addPyFile('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/src/data_creation.py')
sc.addFile('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/config/config.ini')

sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MODBUS_feature_extractor.h5")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MODBUS_LogReg_model.pkl")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MODBUS_Random-Forest_model.pkl")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MODBUS_SDAE_model.h5")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MODBUS_SDAE_model.h5")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MQTT_feature_extractor.h5")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MQTT_KNN_model.pkl")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MQTT_LogReg_model.pkl")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MQTT_Random-Forest_model.pkl")
sc.addFile("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/MQTT_SDAE_model.h5")
'''
sc.setSystemProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

spark = SparkSession.builder.config(conf=conf).getOrCreate()
print(spark.sparkContext.getConf().get("spark.serializer"))


#Now you can shutdown the server by calling this function:
@app.errorhandler(400)
def request_timeout_error(e):
    try:

        request(Timeout=0.0001)
    except:
        return jsonify({"error": "Connection Error", }), 400


@app.errorhandler(500)
def internal_server_error(e):
    # note that we set the 500 status explicitly
    return render_template(os.path.join(os.getcwd(), "/templates/500.html")), 500


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from self-learning app!!</h1>"


@app.route("/userinteraction", methods=['POST'])
def TrainingModels():
    # Get the netflow_id and different label by user in order to be changed in
    # labelled data of each protocol in mongodb
    request_body = request.get_json()
    #request_body = initial_request[0]
    print(request_body)
    print(json)
    protocols = []
    netflow_ids = []
   
    for item in request_body['data']:
        ##netflow_ids.append(item['value'])
        netflow_ids.append(item['id'])

    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    
    x = pd.DataFrame()
    y = pd.DataFrame()

    mqtt_labels = []
    modbus_labels = []
    modbus_initial= pd.DataFrame(list(modbus.find()))
    print('==========================')
    print(modbus_initial.iloc[-1]['Timestamp'])
    mqtt_initial = pd.DataFrame(list(mqtt.find()))
    now_modbus = int(time.mktime(datetime.datetime.strptime(modbus_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p').timetuple()))
    down_modbus = int(time.mktime((datetime.datetime.strptime(modbus_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p') - datetime.timedelta(hours = 1)).timetuple()))
    up_modbus = int(time.mktime((datetime.datetime.strptime(modbus_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p') + datetime.timedelta(hours = 3)).timetuple()))
    now_mqtt = int(time.mktime(datetime.datetime.strptime(mqtt_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p').timetuple()))
    down_mqtt = int(time.mktime((datetime.datetime.strptime(mqtt_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p') - datetime.timedelta(hours = 1)).timetuple()))
    up_mqtt = int(time.mktime((datetime.datetime.strptime(mqtt_initial.iloc[-1]['Timestamp'], '%d/%m/%Y %I:%M:%S %p') + datetime.timedelta(hours = 3)).timetuple()))
    # the subsets of database records, which correspond to user interaction,
    # are converted to pandas dataframe
    ##for item in request_body:
    for item in request_body['data']:
        print('====================')
        print(item['id'].split('-'))
        print('====================')
        seq = item['id'].split('-')
        x = pd.concat([x, pd.DataFrame(list(modbus.find({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_modbus,"$lt":up_modbus}})))]).reset_index(drop=True)
        #y = pd.concat([y, pd.DataFrame(list(mqtt.find({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}]})))]).reset_index(drop=True)
        y = pd.concat([y, pd.DataFrame(list(mqtt.find({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_mqtt,"$lt":up_mqtt}})))]).reset_index(drop=True)
        
        if x.empty:
            pass
        else:
            
            print(x['Timestamp'])           
            protocol = 'modbus'
            print(protocol)
            print('===============================')
            protocols.append('MODBUS')
            modbus.update_many({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_modbus,"$lt":up_modbus}},
                               {
                                   "$set": {"Label": item['label']},
                                   # $currentDate: { lastModified: true }
                               })
            x1 = pd.DataFrame(list(modbus.find({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_modbus,"$lt":up_modbus}}))).reset_index(drop=True)
            if x1.empty:
                pass
            else:
                modbus_labels.append(x1.iloc[-1]['Label'])
            
            # We make 20 copies of the same netflow in order to be avoided problems due to
            # lack of data in training process
           
            data_object = data_creation(protocol, db)
            for i in range(0, len(x1)):
                    # convert_to_np module avoid some bugs we face in restoring data to mongo db
                    # collections
                    data_object.convert_to_np(db["history_modbus"], x1.iloc[i, 1:].to_dict())
       
        if y.empty:
            pass
        else:
            protocol = 'mqtt'
            # protocol = 'MODBUS'
            print(y['Timestamp'])
            print(protocol)
            print('====================================')  
            protocols.append('MQTT')
            mqtt.update_many(
                {"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_mqtt,"$lt":up_mqtt}},
                    {
                        "$set": {"Label": item['label']},
                    }
                )
            y1 = pd.DataFrame(list(mqtt.find({"$or": [{"Src IP":seq[0], "Dst IP":seq[1]}, {"Dst IP":seq[0],"Src IP":seq[1]}], "timestamp": { 
            "$gt": down_mqtt,"$lt":up_mqtt}}))).reset_index(drop=True)
            if y1.empty:
                pass
            else:
                mqtt_labels.append(y1.iloc[-1]['Label'])
           
                
                
            data_object = data_creation(protocol, db)
            for i in range(0, len(y1)):
                    data_object.convert_to_np(db["history_mqtt"], y1.iloc[i, 1:].to_dict())
                    #data_object.convert_to_np(db["recent_history_μ;ττ"], y1.iloc[i, 1:].to_dict())

    list_set_mqtt = set(mqtt_labels)
    # convert the set to the list
    mqtt_labels = (list(list_set_mqtt))
    list_set_modbus = set(modbus_labels)
    # convert the set to the list
    modbus_labels = (list(list_set_modbus))
    list_set = set(protocols)
    # convert the set to the list
    protocols = (list(list_set))
    mqtt_attacks = db['mqtt_attacks']
    modbus_attacks = db['modbus_attacks']

    for protocol in protocols:
        modelTypes = config['Model_types']['model_types'].split(',')
        crossVal = int(config['Model_types']['crossval'])
        protocols = config['Protocols']['protocols'].split(',')
        all_models = ['KNN','SDAE','Random-Forest','LogReg']
        protocol = protocol  # request_body['protocol']
        if protocol in protocols:
            pass
        else:
            return render_template(os.path.join(os.getcwd(), "/templates/400_protocol.html")), 500
            

        if (modelTypes in all_models) or (set(modelTypes).issubset(set(all_models))):
            pass
        else:
            return render_template(os.path.join(os.getcwd(), "/templates/400_model.html")), 500
            
        
        if protocol == 'MQTT':
            attackLabelsToInt = {
                "Label": {"Normal": 0, "Unauthorized Subscribe": 1, "Large Payload": 2, "Connection Overflow": 3}}
            train_trigger(protocol = protocol,attackdb = mqtt_attacks, labels = mqtt_labels, attackLabelsToInt = attackLabelsToInt, db=db )
        if protocol == 'MODBUS':

            attackLabelsToInt = {
                "Label": {'Normal': 0, 'UID brute force': 1, 'Enumeration Function': 2,
                          'Fuzzing Read Holding Registers': 3}}
            train_trigger(protocol=protocol, attackdb= modbus_attacks, labels=modbus_labels,
                          attackLabelsToInt=attackLabelsToInt, db=db)
    
   
    
    return 'database is updated and training process has been started', 200
    #return 'training process has been finished', 200

def train_trigger(protocol, attackdb, labels, attackLabelsToInt, db):
        # For each protocol, it is took place the corresponding process
        #process critical is created to be decided the module; if process_critical = 0 retraining module is activated and if process_critical = 1 train from scratch module is activated
        process_critical = 0
        models = config['Model_types']['model_types'].split(',')
        cv = int(config['Model_types']['crossval'])
        for attack in labels:
            # 1st: case mqtt_attacks is in mongodb collections. This means that
            # system works at least one time
            if "{}_attacks" .format(protocol.lower())in db.list_collection_names():
                
                # if the attack is in list of attacks means that it is not needs to retrain
                # the system from scratch. It is updated the deep learning model with new data
                # but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                if attack in list(list(list(attackdb.find())[0].items())[1][1]):
                    pass
                    
                # In case that we have a new attack:
                # 1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                # 2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                # 3. drop the old collection in mongodb that represents the old attack list
                # 4. Restore the new attack list
                # 5. Get the new number of classes
                # Train all models from scratch with TrainModels module. The logits of deep learning
                # model are changed so it is obligable to be made this process.

                else:

                    attack_dict = list(attackdb.find()[0].items())[1][1]
                    attack_dict['{}'.format(attack)] = list(list(attackdb.find()[0].items())[1][1].values())[
                                                               -1] + 1
                    attackLabelsToInt = {"Label": attack_dict}
                    db["{}_attacks".format(protocol.lower())].drop()
                    mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                    num_classes = len(list(list(attackdb.find()[0].items())[1][1].values()))
                    process_critical  += 1             

            else:

                # 3rd case: In this case
                #         a. The system not run in past\
                #         b. The attack belongs to initial attack list
                # In this case it is retrained the initail models of system
                if attack in list(list(attackLabelsToInt.values())[0].keys()):
                    pass
                   
                # 4th case: In this case
                #          a. the system not run in past
                #          b. the attack is totally new and models are trained from scratch.
                else:
                    attack_dict = list(attackLabelsToInt.values())[0]
                    attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1]) + 1
                    attackLabelsToInt = {"Label": attack_dict}
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                    process_critical += 1
                    
        if process_critical == 0:
            mode = 'Retraining'
            if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                num_classes = len(list(list(attackdb.find()[0].items())[1][1].values()))
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                target_names = list(list(db["{}_attacks".format(protocol.lower())].find()[0].items())[1][1].values())
                Retrain(protocol, attackLabelsToInt, num_classes, target_names, mode)
            else:

                mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                target_names = list(list(db["{}_attacks".format(protocol.lower())].find()[0].items())[1][1].values())
                Retrain(protocol, attackLabelsToInt, num_classes, target_names, mode)
        else:
            if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                num_classes = len(list(list(attackdb.find()[0].items())[1][1].values()))
                target_names = list(list(db["{}_attacks".format(protocol.lower())].find()[0].items())[1][1].values())
                trainFromScratch(protocol, models, cv, num_classes, attackLabelsToInt, target_names)
            else:
                
                
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                target_names =list(list(attackLabelsToInt.values())[0].keys())
                target_names.append(attack)
                num_classes = len(target_names)
                trainFromScratch(protocol, models, cv, num_classes, attackLabelsToInt, target_names)
           
def Sort_Tuple(tup): 
      
    # getting length of list of tuples
    lst = len(tup) 
    for i in range(0, lst): 
          
        for j in range(0, lst-i-1): 
            if (tup[j][1] > tup[j + 1][1]): 
                temp = tup[j] 
                tup[j]= tup[j + 1] 
                tup[j + 1]= temp 
    return tup 

def trainFromScratch(protocol, models, cv, num_classes, attackLabelsToInt, target_names):
    data_object = data_creation(protocol, db)
    data = data_object.create_data(attackLabelsToInt)
    register_spark()
    start_time = time.time()
    data = Data(data, protocol)
    x_train, x_test, y_train, y_test = data.train_test_data()
    results = Parallel(backend='spark',n_jobs= len(models) )(
        delayed(parallelTrain)(x_train, y_train, x_test, y_test,data, protocol, model, cv, num_classes, attackLabelsToInt, target_names) for model in models)
    print(results)
    best_f1 = 0
    best_model = 'some tuples are None'
    for tup in results:
        if tup != None:
            if tup[1] != np.nan:
                if tup[1] > best_f1:
                    best_f1 = tup[1]
                    best_model: object = tup[0]
            
            if tup[2] == config['server1']['hostname']:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(config['server1']['ip_address'],
                username = config['server1']['username'], password = config['server1']['password'], allow_agent = False)
                sftp = ssh.open_sftp()
                sftp.chdir(os.path.join(os.getcwd(), "/models"))
                if tup[0] == 'SDAE':
                    sftp.get(os.path.join(os.getcwd(), "/models/{}_SDAE_model.h5".format(protocol)),os.path.join(os.getcwd(), "/models/{}_SDAE_model.h5".format(protocol)))
                else:
                    sftp.get(os.path.join(os.getcwd(), '{}_{}_model.pkl'.format(protocol, tup[0])),os.path.join(os.getcwd(), '{}_{}_model.pkl'.format(protocol, tup[0])))
    
    print('Best model: ', best_model, ' with best value: ', best_f1)
    
    if best_model == 'SDAE':
        model = load_model(os.path.join(os.getcwd(), '/models') + protocol + "_" + best_model + "_model.h5")
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol) + protocol + "_" + best_model + '_model.h5')
        model.save(os.path.join(os.getcwd(), '/best_models/{}/'.format(protocol) + protocol + "_" + best_model + '_model.h5'))
        search_dir = (os.path.join(os.getcwd(), '/best_models/{}/'.format(protocol)))
        files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
        if len(files) > 1:
            # remove anything from the list that is not a file (directories, symlinks)
            # thanks to J.F. Sebastion for pointing out that the requirement was a list 
            # of files (presumably not including directories)  
            tup_list = []
            for file in files:
                tup = (file,os.path.getmtime(file))
                tup_list.append(tup)
                    
            os.remove(Sort_Tuple(tup_list)[0][0])
    else:
        model = pickle.load(open(os.path.join(os.getcwd(), '/models/' + protocol + "_" + best_model + '_model.pkl'), 'rb'))
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol) + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
            search_dir = '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol)
            files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
            if len(files) > 1:
                # remove anything from the list that is not a file (directories, symlinks)
                # thanks to J.F. Sebastion for pointing out that the requirement was a list 
                # of files (presumably not including directories)  
                tup_list = []
                for file in files:
                    tup = (file,os.path.getmtime(file))
                    tup_list.append(tup)
                os.remove(Sort_Tuple(tup_list)[0][0])
    if "{}_prediction_attacks" .format(protocol.lower())in db.list_collection_names():
        db["{}_prediction_attacks".format(protocol.lower())].drop()
    mycol = db["{}_prediction_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
    db["labelled_modbus"].drop()
    db["labelled_mqtt"].drop()   
    print('Execution time: %s seconds' % (time.time() - start_time))
    

def parallelTrain(x_train, y_train, x_test, y_test,csv, protocol, model_type, cv, num_classes, attackLabelsToInt, target_names):

    hostname = socket. gethostname()
    #local_ip = socket. gethostbyname(hostname)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    if y_train.dtypes == np.float64:
        y_train = y_train.astype({"Label": np.float32})
    if y_train.dtypes == np.int64:
         y_train = y_train.astype({"Label": np.int32})

    if y_test.dtypes == np.float64:
        y_test = y_test.astype({"Label": np.float32})
    if y_test.dtypes == np.int64:
        y_test = y_test.astype({"Label": np.int32})
    #IN THIS POINT IT IS CALLED SCALING MODULE FROM DATA_PROCESS.PY
    train_init = Data(x_train,protocol)
    x_train = train_init.scaling(model_type)
    test_init = Data(x_test,protocol)
    x_test = test_init.scaling(model_type)
    mode = 'TrainingFromScratch'
    train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), csv, protocol, model_type, cv,  num_classes, attackLabelsToInt, target_names, mode, hostname)
    model, report = train.classify()
    if model_type == 'SDAE':
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/' + protocol + "_" + model_type + '_model.h5')
        print("Saved" + protocol + "model to disk")
        
    else:
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/' + protocol + "_" + model_type + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid) 
    print('F1 score for ', model_type, ': ', report.get('weighted avg').get('f1-score'))
    f1 = report.get('macro avg').get('f1-score')
    return model_type, f1, hostname #model



def Retrain(protocol, attackLabelsToInt, num_classes, target_names, mode):
    start_time = time.time()
    # We make the same process as TrainModels for retraining module
    modelTypes = config['Model_types']['model_types'].split(',')
    crossVal = int(config['Model_types']['crossval'])
    register_spark()
    data_object = data_creation(protocol, db)
    data = data_object.create_data(attackLabelsToInt)
    data = Data(data, protocol)
    # It is called the training module
    results = Parallel(backend= 'spark', n_jobs=len(modelTypes))(
       delayed(parallelRetrain)(data,protocol, model, crossVal, attackLabelsToInt, num_classes, target_names, mode) for model in modelTypes)#, db 
    best_f1 = 0
    print('best f1 is')
    print(best_f1)
    print(results)
    for tup in results:
        if tup[1] > best_f1 and tup != np.nan:
            best_f1 = tup[1]
            best_model: object = tup[0]
        
        #if tup[2] == 'iti-572':
        if config['server1']['hostname']:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(config['server1']['ip_address'],
                username = config['server1']['username'], password = config['server1']['password'], allow_agent = False)
                sftp = ssh.open_sftp()
                sftp.chdir('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models')
                if tup[0] == 'SDAE':
                    sftp.get('{}_SDAE_model.h5'.format(protocol),'/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/{}_SDAE_model.h5'.format(protocol) )
                else:
                    sftp.get('{}_{}_model.pkl'.format(protocol, tup[0]),'/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/{}_{}_model.pkl'.format(protocol, tup[0])) 
                    
       
    print('Best model: ', best_model, ' with best value: ', best_f1)
    print('Execution time of model %s seconds' % (time.time() - start_time))
    if best_model == 'SDAE':
        model = load_model("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/" + protocol + "_" + best_model + "_model.h5")
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol) + protocol + "_" + best_model + '_model.h5')
        search_dir = '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol)
        files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
        if len(files) > 1:
            # remove anything from the list that is not a file (directories, symlinks)
            # thanks to J.F. Sebastion for pointing out that the requirement was a list 
            # of files (presumably not including directories)  
            tup_list = []
            for file in files:
                tup = (file,os.path.getmtime(file))
                tup_list.append(tup)
            os.remove(Sort_Tuple(tup_list)[0][0])
    else:
        
        model = pickle.load(open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/' + protocol + "_" + best_model + '_model.pkl', 'rb'))
        
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol) + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
            search_dir = '/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/best_models/{}/'.format(protocol)
            files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
            if len(files) > 1:
                # remove anything from the list that is not a file (directories, symlinks)
                # thanks to J.F. Sebastion for pointing out that the requirement was a list 
                # of files (presumably not including directories)   
                tup_list = []
                for file in files:
                    tup = (file,os.path.getmtime(file))
                    tup_list.append(tup)
                os.remove(Sort_Tuple(tup_list)[0][0])
    if "{}_prediction_attacks" .format(protocol.lower())in db.list_collection_names():
        db["{}_prediction_attacks".format(protocol.lower())].drop()
    mycol = db["{}_prediction_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
    db["labelled_modbus"].drop()
    db["labelled_mqtt"].drop()       
    print('Execution time: %s seconds' % (time.time() - start_time))
    return 'Model re-training  started'

def parallelRetrain(data,protocol, model_type, cv, attackLabelsToInt,  num_classes, target_names, mode):#attack,, db
    myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    db = myclient["labelled_netflows_db"]
    if model_type == "SDAE" and mode == 'Retraining':#or model_type == "DNN"
        data_object = data_creation(protocol, db)
        short_data = Data(data_object.create_short_data(attackLabelsToInt,model_type), protocol)
        # It is called the training module
        x_train, x_test, y_train, y_test = short_data.train_test_data()
        train_init = Data(x_train,protocol)
        x_train = train_init.scaling(model_type)
        test_init = Data(x_test,protocol)
        x_test = test_init.scaling(model_type)
    else:
        
        x_train, x_test, y_train, y_test = data.train_test_data()
        #IN THIS POINT IT IS CALLED SCALING MODULE FROM DATA_PROCESS.PY
        train_init = Data(x_train,protocol)
        x_train = train_init.scaling(model_type)
        test_init = Data(x_test,protocol)
        x_test = test_init.scaling(model_type)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if y_train.dtypes == np.float64:
        y_train = y_train.astype({"Label": np.float32})
    if y_train.dtypes == np.int64:
        y_train = y_train.astype({"Label": np.int32})
    if y_test.dtypes == np.float64:
        y_test = y_test.astype({"Label": np.float32})
    if y_test.dtypes == np.int64:
        y_test = y_test.astype({"Label": np.int32})
    
    hostname = socket. gethostname()
    
    train =Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv,  num_classes, attackLabelsToInt, target_names, mode, hostname)    
    model, report = train.classify()
    
    if model_type == 'SDAE':
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/' + protocol + "_" + model_type + '_model.h5')
        print("Saved" + protocol + "model to disk")
    else:
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/models/' + protocol + "_" + model_type + '_model.pkl', 'wb') as fid:
                    pickle.dump(model, fid)   
    print('F1 score for ', model_type, ': ', report.get('weighted avg').get('f1-score'))
    f1 = report.get('macro avg').get('f1-score')
    return model_type, f1, hostname

    


if __name__ == '__main__':
    #CORS(app.app)
    app.run(host='160.40.53.119', port=5002, ssl_context=context)
     








    







