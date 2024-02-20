from os import lchown
from flask import Flask, request, jsonify, json, render_template,redirect, url_for
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
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import psutil
tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process
import requests
#import psutil
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
import scheduler_initial
findspark.init()
config = configparser.ConfigParser()
config.read('config.ini')
myclient = pymongo.MongoClient('localhost', 27017)
#myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
db = myclient["labelled_netflows_db"]
#modbus = db["labelled_modbus"]

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
avlb = psutil.virtual_memory().available
print(avlb)
print(type(avlb))
digits = (int)(math.log10(avlb))
 
# Find first digit
first = (int)(avlb/ pow(10, digits))
exec_memory = str(first-1)+'g'
print(exec_memory)
print(type(exec_memory))
app = Flask(__name__)

conf = SparkConf().setAppName('Self-learning_Retrain Framework').setMaster('spark://160.40.49.209:7077').set('spark.executor.memory', exec_memory)#.set('spark.memory.fraction', 0.3).set('spark.memory.storageFraction',0.3)
sc = SparkContext(conf=conf)
print(sc.version)
print(sc._jsc.version())

sc.addPyFile('data_process.py')
sc.addPyFile('training.py')
sc.addPyFile('SDA.py')
sc.addPyFile('SDA_retrain.py')
sc.addPyFile('retraining.py')
sc.addPyFile('training.py')
sc.addFile('config.ini')
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
    return render_template('500.html'), 500


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from self-learning app!!</h1>"


# After user select the items that he/she is consider that has wrong labels, we restore
# the items in a specific database collection eg new_attacks_mqtt in order to
# be get in next stages
@app.route("/PostSelectedNetflow", methods=['POST'])
def post_request():
    request_body = request.get_json()
    print(request_body)
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    

    ##db = myclient["labelled_netflows_db"]
    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    backnet = db['labelled_backnet']
    ntp = db['labelled_ntp']
    radius = db['labelled_radiys']
    bacnet = db['labelled_bacnet']
    
    for item in request_body:
        print(item)
        x = pd.DataFrame(list(modbus.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))
        y = pd.DataFrame(list(mqtt.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))
        z = pd.DataFrame(list(ntp.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))
        k = pd.DataFrame(list(radius.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))
        n = pd.DataFrame(list(bacnet.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))
        if x.empty:
            pass
        else:
            protocol = 'MODBUS'
            mycol = db["new_attacks_modbus"].insert_one(item)
        if y.empty:
            pass
        else:
            protocol = 'MQTT'
            mycol = db["new_attacks_mqtt"].insert_one(item)
        if z.empty:
            pass
        else:
            protocol = 'NTP'
            mycol = db["new_attacks_ntp"].insert_one(item)
        if k.empty:
            pass
        else:
            protocol = 'RADIUS'
            mycol = db["new_attacks_radius"].insert_one(item)
        if n.empty:
            pass
        else:
            protocol = 'BACNET'
            mycol = db["new_attacks_bacnet"].insert_one(item)
    
    return 'Selected netflow is restored', 200


# Get the netflow_ids in string format in order to be retrained the models in next stages
@app.route("/GetSelectedNetflow", methods=['GET'])
def get_request():
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    
    ##db = myclient["labelled_netflows_db"]
    # db.tutorial.find()
    labeled = []
    protocols = ['MODBUS', 'MQTT', 'NTP''RADIUS', 'BACNET']
    for protocol in protocols:
        labeled_db = db["new_attacks_{}".format(protocol.lower())]
        labeled_med = labeled_db.find()
        for item in labeled_med:
            if (type(item)) == dict:
                labeled.append(item['value'])
    #print(str(labeled))
    return labeled  # 'Get the user requests'


@app.route("/GetUserInteraction", methods=['POST'])
def TrainingModelsIter():
    # Get the netflow_id and different label by user in order to be changed in
    # labelled data of each protocol in mongodb
    request_body = request.get_json()
    protocols = []
    netflow_ids = []
   
    for item in request_body:
        netflow_ids.append(item['value'])
        # labels.append(item['Label'])

    
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    
    ##db = myclient["labelled_netflows_db"]

    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    print(modbus.find())
    print(mqtt.find())

   

    x = pd.DataFrame()
    y = pd.DataFrame()

    mqtt_labels = []
    modbus_labels = []

    # the subsets of database records, which correspond to user interaction,
    # are converted to pandas dataframe
    for item in request_body:
        print('ITEM IS &&&&&&&&&&&&&')
        print(item)
        netflow_ids.append(item['value'])
        
        x = pd.concat([x, pd.DataFrame(list(modbus.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))]).reset_index(drop=True)
        y = pd.concat([y, pd.DataFrame(list(mqtt.find({"Flow ID": item['value'], "timestamp": item['timestamp']})))]).reset_index(drop=True)


        if x.empty:
            pass
        else:
            # the labels in labelled data in database are changed according to
            # new labels are set by users.
            
            
           
            protocols.append('MODBUS')

            modbus.update_many({"Flow ID": item['value'], "timestamp": item['timestamp'] },
                               {
                                   "$set": {"Label": item['Label']},
                                   # $currentDate: { lastModified: true }
                               })
            # because we want the updated data, it is converted the new data in pandas datafram
            modbus_labels.append(x.iloc[-1]['Label'])
            print('----------MODBUS LABELS------------------')
            print(modbus_labels)
            x = pd.DataFrame(list(modbus.find({"Flow ID": item['value'], "timestamp": item['timestamp']}))).reset_index(drop=True)

            # We make 20 copies of the same netflow in order to be avoided problems due to
            # lack of data in training process
            #####for j in range (0, 20):
            for j in range(0, 20):
                # We have two types of collections:
                # 1. retrain_modbus: the current changes by user. This collection is deleted
                # after training process
                # 2. history_mqtt: the change history of users. In this case we have all changes
                # by the strat of system.
                for i in range(0, len(x)):
                    # convert_to_np module avoid some bugs we face in restoring data to mongo bd
                    # collections
                    #convert_to_np(db["retrain_modbus"], x.iloc[i, 1:].to_dict())
                    convert_to_np(db["history_modbus"], x.iloc[i, 1:].to_dict())
       
        if y.empty:
            pass
        else:
            # protocol = 'MQTT'
            # protocol = 'MODBUS'
            
            protocols.append('MQTT')
            mqtt.update_many(
                {"Flow ID": item['value'], "timestamp": item['timestamp']},
                {
                    "$set": {"Label": item['Label']},
                    # $currentDate: { lastModified: true }
                }
            )
            y = pd.DataFrame(list(mqtt.find({"Flow ID": item['value'], "timestamp": item['timestamp']}))).reset_index(drop=True)
            mqtt_labels.append(y.iloc[-1]['Label'])
            for j in range(0, 20):
                for i in range(0, len(y)):
                    #convert_to_np(db["retrain_mqtt"], y.iloc[i, 1:].to_dict())
                    convert_to_np(db["history_mqtt"], y.iloc[i, 1:].to_dict())

    list_set_mqtt = set(mqtt_labels)
    # convert the set to the list
    mqtt_labels = (list(list_set_mqtt))
    print('mqtt_labels')
    print(mqtt_labels)
    list_set_modbus = set(modbus_labels)
    # convert the set to the list
    modbus_labels = (list(list_set_modbus))
    list_set = set(protocols)
    # convert the set to the list
    protocols = (list(list_set))
    print(protocols)
    mqtt_attacks = db['mqtt_attacks']
    modbus_attacks = db['modbus_attacks']

    for protocol in protocols:
        
        if protocol == 'MQTT':
            attackLabelsToInt = {
                "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
            train_trigger(protocol = protocol,attackdb = mqtt_attacks, labels = mqtt_labels, attackLabelsToInt = attackLabelsToInt, db=db )
        if protocol == 'MODBUS':

            attackLabelsToInt = {
                "Label": {'Normal': 0, 'UID brute force': 1, 'Enumeration Function': 2,
                          'Fuzzing Read Holding Registers': 3}}
            train_trigger(protocol=protocol, attackdb= modbus_attacks, labels=modbus_labels,
                          attackLabelsToInt=attackLabelsToInt, db=db)
    shutdown_server() 
    
    return 'database is updated and training process has been started', 200

def train_trigger(protocol, attackdb, labels, attackLabelsToInt, db):
        # For each protocol, it is took place the corresponding process
        cnt = 0
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
                    print('list attacks')
                    print(attackLabelsToInt)
                    db["{}_attacks".format(protocol.lower())].drop()
                    mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                    #print('list attacks')
                    #print(list(db["{}_attacks".format(protocol.lower())].find()))
                    num_classes = len(list(list(attackdb.find()[0].items())[1][1].values()))
                    cnt += 1
                    



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
                    print(attackLabelsToInt)
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                    cnt += 1
                    
        if cnt == 0:
            print('This is 0')
            if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                num_classes = len(list(list(attackdb.find()[0].items())[1][1].values()))
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                retrainModels(protocol=protocol, attackLabelsToInt=attackLabelsToInt)
            else:

                mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                print('RETRAIN MODULE')
                retrainModels(protocol=protocol, attackLabelsToInt=attackLabelsToInt)
        else:
            if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
                # pass
                attackLabelsToInt = {"Label": list(list(attackdb.find())[0].items())[1][1]}
                print('*******************************')
                print(attackLabelsToInt)
                TrainModels(protocol=protocol, attackLabelsToInt=attackLabelsToInt, attack=attack, db = db)
            else:
                print('TRAIN MODULE')
                # pass
                TrainModels(protocol=protocol, attackLabelsToInt=attackLabelsToInt, attack=attack, db=db)

def convert_to_np(collection, r):
    try:
        collection.insert(r)
    except pymongo.errors.InvalidDocument:
        # Python 2.7.10 on Windows and Pymongo are not forgiving
        # If you have foreign data types you have to convert them
        n = {}
        for k, v in r.items():

            if isinstance(v, np.int64):
                v = int(v)
            #if isinstance(v, np.float64):
                #v = np.float32(v)

            n[k] = v

        collection.insert(n)


def TrainModels(protocol, attackLabelsToInt, attack, db):
    # As we make the same training process we hard-coded the request
    # for training in parallel process
    protocol = protocol
    #modelTypes= ["LogReg","SDAE", "KNN","Random-Forest"]#"SDAE", "KNN","LogReg", "Random-Forest"
    modelTypes= ["Random-Forest"]
    crossVal = 2

    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    ##db = myclient["labelled_netflows_db"]
    start = time.time()
    request_body = request.get_json()
    protocols = config['Protocols']['protocols'].split(',')
    all_models = config['Model_types']['model_types'].split(',')
    protocol = protocol  # request_body['protocol']
    # error handling in case that protocols and ModelType is not in predifined
    if protocol in protocols:
        pass
    else:
        return jsonify({"error": "Invalid Requested Protocol", }), 400

    if (modelTypes in all_models) or (set(modelTypes).issubset(set(all_models))):
        pass
    else:
        return jsonify({"error": "Invalid Requested Model", }), 400

    end = time.time()

    train(create_data(protocol,  attackLabelsToInt, db), protocol, modelTypes, crossVal, attack)


    return 'Model re-training  started'


def retrainModels(protocol, attackLabelsToInt):
    protocol = protocol
    #modelTypes= ["LogReg","SDAE", "KNN","Random-Forest"]#"SDAE", "Random-Forest",
    modelTypes= ["Random-Forest"]
    crossVal = 2 
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    ##db = myclient["labelled_netflows_db"]
    start = time.time()
    request_body = request.get_json()
    protocols = config['Protocols']['protocols'].split(',')
    all_models = config['Model_types']['model_types'].split(',')
    protocol = protocol  # request_body['protocol']

    if protocol in protocols:
        pass
    else:
        return jsonify({"error": "Invalid Requested Protocol", }), 400

    if (modelTypes in all_models) or (set(modelTypes).issubset(set(all_models))):
        pass
    else:
        return jsonify({"error": "Invalid Requested Model", }), 400
    end = time.time()

    # We make the same process as TrainModels for retraining module
    ##train_task = Process(target=Retrain, args=(protocol, modelTypes, crossVal, attackLabelsToInt))
    
    Retrain(protocol, modelTypes, crossVal, attackLabelsToInt)

    return 'Model re-training  started'


#def create_data(protocol, model, attackLabelsToInt, db):
def create_data(protocol,  attackLabelsToInt, db):
    # It is concated the data points from csvs and the data points from historic data
    # of historic collection
    # initial csv is read
    #IN THIS CASE THE NAME OF CSVS IS CHANGED AS TO BE OPTIMIZED THE CODE
    csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/datasets/{}_dataset.csv".format(protocol.lower()))
    # 1. we take into account the historical data from user interactions from the
    # past and now.
    history = db["history_{}".format(protocol.lower())]
    label = db["labelled_{}".format(protocol.lower())]
    label_df = pd.DataFrame(list(label.find())).iloc[:, 1:]
    history_df = pd.DataFrame(list(history.find())).iloc[:, 1:]
    all_db_df = pd.concat([history_df, label_df]).reset_index(drop=True)
    columns = list(set(all_db_df.columns).intersection(csv.columns))
    columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
    csv  = csv[columns]
    all_db_df = all_db_df[columns]
    csv = pd.concat([csv , all_db_df]).reset_index(drop=True)
    csv.replace(attackLabelsToInt, inplace=True)
    columns = csv.columns
    print(columns)
    print(type(columns))
    for column in columns:
        print(column)
        if type(csv.iloc[0][column]) == np.float64:
            csv = csv.astype({"{}".format(column): np.float32})
        if type(csv.iloc[0][column]) == np.int64:
            csv = csv.astype({"{}".format(column): np.int32})

    print('======================================')
    print('after transformation')
    print(csv)
    print(csv['Label'].unique())
    return csv
    


def train(data, protocol, models, cv, attack):
    # register SPARK backend
    register_spark()
    start_time = time.time()
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    ##db = myclient["labelled_netflows_db"]
    csv = data.copy()
    #data = create_data(protocol, attackLabelsToInt, db)
    print('INSIDE TRAIN')
    print(data)
    data = Data(data, protocol)
    print(data)
    # It is called the training module
    x_train, x_test, y_train, y_test = data.train_test_data()
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
    avlb = psutil.virtual_memory().available
    print(avlb)
    print(type(avlb))
    #n_jobs=  2 len(models)
    results = Parallel(backend='spark',n_jobs= len(models) )(
        delayed(parallelTrain)(x_train, y_train, x_test, y_test,data, protocol, model, cv, attack) for model in models)
    best_f1 = 0
    best_model = 'some tuples are None'
    for tup in results:
        if tup != None:
            if tup[1] != np.nan:
                if tup[1] > best_f1:
                    best_f1 = tup[1]
                    best_model: object = tup[0]
    print('Best model: ', best_model, ' with best value: ', best_f1)
    if best_model == 'SDAE':
        model = load_model("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/models/" + protocol + "_" + best_model + "_model.h5")
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/best_models/' + protocol + "_" + best_model + '_model.h5')
    else:
        model = pickle.load(open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/models/' + protocol + "_" + best_model + '_model.pkl', 'rb'))
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/best_models/' + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
    print('Execution time: %s seconds' % (time.time() - start_time))
    #stop()



def scaling(X, protocol, model_type):
    scaler = StandardScaler()
    X = scaler.fit(X).transform(X)
    with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/scalers/' + str(protocol) + "_" + str(model_type) + '_scaler.pkl', 'wb')as fid:
        pickle.dump(scaler, fid)

    print('scaler for ', str(protocol),' exists')
    #load the scaler
    scaler = load(open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/scalers/' + str(protocol) + "_" + str(model_type) +'_scaler.pkl', 'rb'))
    #scaler = load(open('scalers/' + self.protocol + "_" + self.model_type +'_scaler.pkl', 'rb'))
    X = scaler.fit(X).transform(X)
    return X
def parallelTrain(x_train, y_train, x_test, y_test,csv, protocol, model_type, cv, attack):
#def parallelTrain(csv, protocol, model_type, cv, attack):
    myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    db = myclient["labelled_netflows_db"]
    # if model_type != 'SDAE':
    # pass
    # else:
    x_train = scaling(x_train, protocol, model_type)
    x_test = scaling(x_test, protocol, model_type)
    train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), csv, protocol, model_type, cv, db, attack)#, db, attack
    report = train.classify()
    print('F1 score for ', model_type, ': ', report.get('weighted avg').get('f1-score'))
    f1 = report.get('macro avg').get('f1-score')
    return model_type, f1

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Shutdown Server....'    
def Retrain(protocol, models, cv, attackLabelsToInt):
    start_time = time.time()
    ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    ##db = myclient["labelled_netflows_db"]
    register_spark()
    data = create_data(protocol, attackLabelsToInt, db)
    data = Data(data, protocol)
    # It is called the training module
    x_train, x_test, y_train, y_test = data.train_test_data()
    #n_jobs=len(models) 2
    results = Parallel(backend= 'spark', n_jobs=len(models))(
        delayed(parallelRetrain)(x_train, y_train, x_test, y_test,data, protocol, model, cv, attackLabelsToInt) for model in models)
    #pool = ThreadPool(len(models))
    #for model in models:
        #pool.map((parallelRetrain(x_train, y_train, x_test, y_test,data, protocol, model, cv, attackLabelsToInt) ), range(len(models)))
    best_f1 = 0
    print('best f1 is')
    print(best_f1)
    print(results)
    for tup in results:
        if tup[1] > best_f1 and tup != np.nan:
            best_f1 = tup[1]
            best_model: object = tup[0]
    print('Best model: ', best_model, ' with best value: ', best_f1)
    print('Execution time of model %s seconds' % (time.time() - start_time))
    if best_model == 'SDAE':
        print('best  models is DL')

        model = tf.keras.models.load_model("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/models/" + protocol + "_" + best_model + "_model.h5")
        model.save('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/best_models/' + protocol + "_" + best_model + '_model.h5')

    else:
        print('best  models is ML')
        model = pickle.load(open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/models/' + protocol + "_" + best_model + '_model.pkl', 'rb'))
        with open('/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/best_models/' + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
    
    

def parallelRetrain(x_train, y_train, x_test, y_test,csv, protocol, model_type, cv, attackLabelsToInt):#attack
    myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
    db = myclient["labelled_netflows_db"]
    # In this function we have some special steps
    # 1. we take into account the historical data from user interactions from the
    # past and now.
    # 2. We get some random samples from initail csv as to have the same number of records
    # for each attack
    # 3. Concatenate the historic interactions in pandas dataframe format with random samples
    # of initial csvs.
    # 4. We get the columns that is the same from initial csvs and the collection historic data.
    # 5. We have 76 features
    if model_type == "SDAE" or model_type == "DNN":
        init_csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self-learning_retrain/datasets/{}_dataset.csv".format(protocol.lower()))
        # len_attacks = create_data(protocol, model_type, attackLabelsToInt)#[1]

        attack_list = []

        for k in attackLabelsToInt['Label']:
            attack_list.append(k)

        counter = 0
        limit = 64  # -len_attacks
        print('inside parallel retrain')
        final_df = pd.DataFrame(columns=init_csv.columns)
        print(limit)
        while final_df.empty:
            for item in attack_list:
                # 2. We get some random samples from initail csv as to have the same number of records
                # for each attack
                med_df = (init_csv[init_csv['Label'] == item])
                if len(med_df) > int(limit / len(attack_list)):
                    new_df = med_df.sample(n=int(limit / len(attack_list)))
                else:
                    new_df = med_df.copy()
                # 4. We get the columns that is the same from initial csvs and the collection historic data.
                columns = list(set(final_df.columns).intersection(init_csv.columns))
                #columns = list(set(all_db_df.columns).intersection(csv.columns))
                columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
                new_df = new_df[columns]

                final_df = pd.concat([final_df, new_df])
                #print('final_df_is')
                #print(final_df)
        print('out of first while loop')

        list_indices = final_df.index.values.tolist()
        # We get some other random samples to have at least 64 data points.
        # Otherwise we have bug in retraining process
        other_df = pd.DataFrame()
        while other_df.empty:
            if len(final_df) < 64:
                limit = 64 - len(final_df)
                other_df = init_csv.sample(n=limit)
                columns = list(set(other_df.columns).intersection(final_df.columns))
                columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
                other_df = other_df[columns]  # .reset_index(drop=True)
                other_df = other_df.dropna()
            else:
                break
        print('out of second loop')

        other_list_indices = other_df.index.values.tolist()
        final_list_indices = list_indices + other_list_indices
        final_df = pd.concat([final_df, other_df])
        print('THIS IS DF FROM CSVS')
        print(final_df)
        ##1. we take into account the historical data from user interactions from the
        # past and now.
        # x = pd.DataFrame(list(modbus.find({"Flow ID":netflow_id})))
        #train_db = pd.DataFrame(list(db["history_{}".format(protocol.lower())].find()))
        if "history_{}".format(protocol.lower()) in db.list_collection_names() and "labelled_{}".format(protocol.lower()) in db.list_collection_names():
        	train_db = pd.concat([pd.DataFrame(list(db["history_{}".format(protocol.lower())].find())),pd.DataFrame(list(db["labelled_{}".format(protocol.lower())].find()))]).reset_index(drop = True)
        elif "history_{}".format(protocol.lower()) in db.list_collection_names():
        	train_db = pd.DataFrame(list(db["history_{}".format(protocol.lower())].find()))
        else:
         	train_db = pd.DataFrame()
        
        print('train_db')
        print(train_db)
        # train_db = create_data(protocol, model_type, attackLabelsToInt)[0]#.iloc[-len_attacks:, :]
        final_columns = list(set(train_db.columns).intersection(final_df.columns))
        final_columns =  [col for col in final_columns if col != 'Label' ] + ['Label'] 
        train_db = train_db[final_columns]
        print('THIS IS DB FROM COLLECTIONS')
        print(train_db)
        # 4. We get the columns that is the same from initial csvs and the collection historic data.
        final_df = pd.concat([final_df, train_db])
        final_df = final_df.reset_index(drop=True)
        if "{}_attacks".format(protocol.lower()) in db.list_collection_names():
            attackLabelsToInt = list(attackLabelsToInt.values())[0]  # pass

        final_df.replace(attackLabelsToInt, inplace=True)
        columns = final_df.columns
        print(columns)
        print(type(columns))
        for column in columns:
            print(column)
            if type(final_df.iloc[0][column]) == np.float64:
                final_df = final_df.astype({"{}".format(column): np.float32})
            if type(final_df.iloc[0][column]) == np.int64:
                final_df = final_df.astype({"{}".format(column): np.int32})

        print('======================================')
        print('after transformation')
        print(final_df)
        ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
        ##db = myclient["labelled_netflows_db"]
        data = Data(final_df, protocol)
        # It is called the training module
        x_train, x_test, y_train, y_test = data.train_test_data()
        x_train = scaling(x_train, protocol, model_type)
        x_test = scaling(x_test, protocol, model_type)
        print(x_train)
        print(x_test)
        train = Training_DL(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv, db)
        # (self, x_train, y_train, x_test, y_test, data, protocol, model_type, cv)

        report = train.classify(db)
        print('protocol:',protocol)
        print('F1 score for ', model_type, ': ', report.get('weighted avg').get('f1-score'))
        f1 = report.get('macro avg').get('f1-score')
        return model_type, f1
        
    else:
        ##myclient = pymongo.MongoClient("mongodb://160.40.49.209:27017/")
        ##db = myclient["labelled_netflows_db"]
        x_train = scaling(x_train, protocol, model_type)
        x_test = scaling(x_test, protocol, model_type)
        train = Training_ML(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), csv, protocol, model_type, cv, db)
        report = train.classify(db)
        print('protocol:',protocol)
        print('F1 score for ', model_type, ': ', report.get('weighted avg').get('f1-score'))
        f1 = report.get('macro avg').get('f1-score')
        
        return model_type, f1

#Now you can shutdown the server by calling this function:


if __name__ == '__main__':

    while ['python','scheduler_initial.py']  in  [process.cmdline() for process in psutil.process_iter()]:
        time.sleep(1)
    else:
        app.run(host='localhost', port=5002)
        scheduler_initial   
        #shutdown_server()
        #requests.get('http://localhost:5002/quit')







