from flask import Flask, request
from joblib import delayed, Parallel
from joblibspark import register_spark
from training import Training
from data_process import Data
import pickle
from keras.models import load_model
from joblib import delayed, Parallel
import time
import pandas as pd
from multiprocessing import Process
from joblib.parallel import Parallel, delayed, parallel_backend
from pyspark.context import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

app = Flask(__name__)

conf = SparkConf().setAppName('Self-learning Framework').setMaster('spark://160.40.49.209:7077').set('spark.executor.memory', '10g')
sc = SparkContext(conf=conf)
sc.addPyFile('data_process.py')
sc.addPyFile('training.py')
sc.addPyFile('SDA.py')

spark = SparkSession.builder.config(conf=conf).getOrCreate()
		
@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from self-learning app!!</h1>"

@app.route("/trainWithCSVFiles", methods=['POST'])
def initialTrain():
    request_body = request.get_json()
    protocol = request_body['protocol']
    modelTypes = request_body['modelTypes']
    crossVal = request_body['crossValidation']
    print(protocol)
    print(modelTypes)
    if protocol == 'MQTT':
        csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
        attackLabelsToInt = {"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
    elif protocol == 'BACNET':
        csv = pd.read_csv("datasets/bacnet_dataset.csv")
        # bacnet dataset
        attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
    elif protocol == 'MODBUS':
        csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
        #modbus dataset
        attackLabelsToInt = {"Label" : {"Normal" : 0, "UID brute force" : 1, "Enumeration Function" : 2, "Fuzzing Read Holding Registers" : 3}}
    elif protocol == 'NTP':
        csv = pd.read_csv("datasets/ntp_dataset_1.csv")
        #ntp_dataset
        attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
    elif protocol == 'RADIUS':
        csv = pd.read_csv("datasets/radius_dataset.csv")
        # Radius_dataset
        attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
    csv.replace(attackLabelsToInt, inplace = True)
    csv.head()
    train_task = Process(target=train, args=(csv, protocol, modelTypes, crossVal))
    train_task.start()
    #train(csv, protocol, modelTypes, crossVal)
    return 'Model training with CSV files started', 200

@app.route("/retrain",methods=['POST'])
def retrainModels():
    request_body = request.get_json()
    protocol = request_body['protocol']
    modelTypes = request_body['modelTypes']
    crossVal = request_body['crossValidation']
    print(protocol)
    print(modelTypes)
    #NEEDS IMPLEMENTATION TO RETRIEVE DATA FROM MONGO DB AND CONVERT TO PANDAS DATAFRAME
    train(data, protocol, modelTypes, crossVal)

def train(data, protocol, models, cv):
    # register SPARK backend
    register_spark()

    start_time = time.time()
    #with parallel_backend('spark') as (ba, _):
    #	Parallel(n_jobs=len(models))(delayed(parallelTrain)(data, protocol, model, cv) for model in models)
    # partial_parallelTrain = partial(parallelTrain, csv=csv, protocol=protocol)
    results = Parallel(backend='spark', n_jobs=len(models))(
        delayed(parallelTrain)(data, protocol, model, cv) for model in models)

    print(results)

    best_f1 = 0
    for tup in results:
        if tup[1] > best_f1:
            best_f1 = tup[1]
            best_model: object = tup[0]
    print('Best model: ', best_model, ' with best value: ', best_f1)
    if best_model == 'SDAE':
        model = load_model("models/" + protocol + "_" + best_model + "_model.h5")
        model.save('best_models/' + protocol + "_" + best_model + '_model.h5')
    else:
        model = pickle.load(open('models/' + protocol + "_" + best_model + '_model.pkl', 'rb'))
        with open('best_models/' + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
    print('Execution time: %s seconds' % (time.time() - start_time))

def parallelTrain(csv, protocol, model_type, cv):
    data = Data(csv, protocol, model_type)
    x_train, x_test, y_train, y_test = data.train_test_data()
    train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv)
    report = train.classify()
    print('F1 score for ', model_type , ': ' ,  report.get('weighted avg').get('f1-score'))
    f1 = report.get('macro avg').get('f1-score')
    return model_type, f1


#def retrieveMongoDB():

app.run(host='localhost')

