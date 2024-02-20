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
config = configparser.ConfigParser()
config.read('config.ini')
app = Flask(__name__)
@app.errorhandler(400)
def request_timeout_error(e):
    try:
        
        request(Timeout = 0.0001)
    except:
        return jsonify({"error": "Connection Error",}), 400

@app.errorhandler(500)
def internal_server_error(e):
    # note that we set the 500 status explicitly
    return render_template('500.html'), 500



@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello from self-learning app!!</h1>"

#After user select the items that he/she is consider that has wrong labels, we restore
#the items in a specific database collection eg new_attacks_mqtt in order to 
# be get in next stages
@app.route("/PostSelectedNetflow", methods=['POST'])
def post_request():
    request_body = request.get_json()
    print(request_body)
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    #netflow_id = request_body['value']

    
    db = myclient["labelled_netflows_db"]
    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    backnet = db['labelled_backnet']
    ntp = db['labelled_ntp']
    radius = db['labelled_radiys']
    bacnet = db['labelled_bacnet']
    for item in request_body:
        print(item)
        x = pd.DataFrame(list(modbus.find({"Flow ID":item['value']})))
        y = pd.DataFrame(list(mqtt.find({"Flow ID":item['value']})))
        z = pd.DataFrame(list(ntp.find({"Flow ID":item['value']})))
        k = pd.DataFrame(list(radius.find({"Flow ID":item['value']})))
        n = pd.DataFrame(list(bacnet.find({"Flow ID":item['value']})))
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
    r'''
    x = pd.DataFrame(list(modbus.find({"Flow ID":netflow_id})))
    y = pd.DataFrame(list(mqtt.find({"Flow ID":netflow_id})))
    z = pd.DataFrame(list(ntp.find({"Flow ID":netflow_id})))
    k = pd.DataFrame(list(radius.find({"Flow ID":netflow_id})))
    n = pd.DataFrame(list(bacnet.find({"Flow ID":netflow_id})))
    if x.empty:
        pass
    else:
        protocol = 'MODBUS'
        mycol = db["new_attacks_modbus"].insert_one(request_body)
    if y.empty:
        pass
    else:
        protocol = 'MQTT'
        mycol = db["new_attacks_mqtt"].insert_one(request_body)
    if z.empty:
        pass
    else:
        protocol = 'NTP'
        mycol = db["new_attacks_ntp"].insert_one(request_body)
    if k.empty:
        pass
    else:
        protocol = 'RADIUS'
        mycol = db["new_attacks_radius"].insert_one(request_body)
    if n.empty:
        pass
    else:
        protocol = 'BACNET'
        mycol = db["new_attacks_bacnet"].insert_one(request_body)
    '''
    return 'Selected netflow is restored', 200
#Get the netflow_ids in string format in order to be retrained the models in next stages   
@app.route("/GetSelectedNetflow", methods=['GET'])
def get_request():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    #db.tutorial.find()
    labeled = ''
    protocols = ['MODBUS','MQTT','NTP''RADIUS','BACNET']
    for protocol in protocols:
        labeled_db = db["new_attacks_{}".format(protocol.lower())]
        labeled_med = labeled_db.find()
        for item in labeled_med:
            if (type(item)) == dict:
                labeled += item['value']+' '
    print(str(labeled))
    return str(labeled)#'Get the user requests'

@app.route("/GetUserInteraction", methods=['POST'])
def get_user_data():
    #Get the netflow_id and different label by user in order to be changed in
    #labelled data of each protocol in mongodb
    request_body = request.get_json()
    protocols = []
    netflow_ids = []
    #labels = []
    for item in request_body:
        netflow_ids.append(item['value'])
        #labels.append(item['Label'])
    
    #netflow_id = request_body['value'] 
    #attack = request_body['Label']
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    
    
    modbus = db["labelled_modbus"]
    mqtt = db["labelled_mqtt"]
    bacnet = db['labelled_backnet']
    ntp = db['labelled_ntp']
    radius = db['labelled_radiys']
    #The list of attacks (old+new) will be restored in special collection of db
    #eg bacnet_attacks. Because of continuous changes of attacks, as it is inserted new attacks,
    #we should know all the attacks for each protocol
    mqtt_attacks = db['mqtt_attacks']
    modbus_attacks = db["modbus_attacks"]
    bacnet_attacks = db['bacnet_attacks']
    ntp_attacks = db['ntp_attacks']
    radius_attacks = db['radius_attacks']
    x = pd.DataFrame()
    y = pd.DataFrame()
    z = pd.DataFrame()
    a = pd.DataFrame()
    b = pd.DataFrame()
    mqtt_labels = []
    modbus_labels = []
    ntp_labels = []
    radius_labels = []
    bacnet_labels = []
    #the subsets of database records, which correspond to user interaction,
    #are converted to pandas dataframe
    for item in request_body:
        netflow_ids.append(item['value'])
        #labels.append(item['Label'])
        x = pd.concat([x,pd.DataFrame(list(modbus.find({"Flow ID":item['value']})))]).reset_index(drop=True)
        y = pd.concat([y,pd.DataFrame(list(mqtt.find({"Flow ID":item['value']})))]).reset_index(drop=True)
        z = pd.concat([z,pd.DataFrame(list(ntp.find({"Flow ID":item['value']})))]).reset_index(drop=True)
        a = pd.concat([a,pd.DataFrame(list(radius.find({"Flow ID":item['value']})))]).reset_index(drop=True)
        b = pd.concat([b,pd.DataFrame(list(bacnet.find({"Flow ID":item['value']})))]).reset_index(drop=True)
        
        x1 = pd.DataFrame(list(modbus.find({"Flow ID":item['value']}))).reset_index(drop=True)
        y1 = pd.DataFrame(list(mqtt.find({"Flow ID":item['value']}))).reset_index(drop=True)
        z1 = pd.DataFrame(list(ntp.find({"Flow ID":item['value']}))).reset_index(drop=True)
        a1 = pd.DataFrame(list(radius.find({"Flow ID":item['value']}))).reset_index(drop=True)
        b1 = pd.DataFrame(list(bacnet.find({"Flow ID":item['value']}))).reset_index(drop=True)
        if x.empty:
            pass
        else:
            #the labels in labelled data in database are changed according to 
            #new labels are set by users.
            #protocol = 'MODBUS'
            protocols.append('MODBUS')
            modbus.update_many({ "Flow ID":item['value'] },
                {
                    "$set": { "Label": item['Label']},
                    #$currentDate: { lastModified: true }
                })
            #because we want the updated data, it is converted the new data in pandas datafram
            ##x = pd.DataFrame(list(modbus.find({"Flow ID":item['value']})))
            x = pd.concat([x,pd.DataFrame(list(modbus.find({"Flow ID":item['value']})))]).reset_index(drop=True)
            
            #We make 20 copies of the same netflow in order to be avoided problems due to 
            #lack of data in training process
            #####for j in range (0, 20):
            for j in range (0, 20):
                #We have two types of collections:
                #1. retrain_modbus: the current changes by user. This collection is deleted
                #after training process
                #2. history_mqtt: the change history of users. In this case we have all changes
                #by the strat of system.
                for i in range(0, len(x)):
                    #convert_to_np module avoid some bugs we face in restoring data to mongo bd
                    #collections
                    convert_to_np(db["retrain_modbus"],x.iloc[i,1:].to_dict())
                    convert_to_np(db["history_modbus"],x.iloc[i,1:].to_dict())
        if x1.empty:
            pass
        else:
            modbus_labels.append(item['Label'])
        #The same process is remade for the rest of protocols   
        if y.empty:
            pass
        else:
            #protocol = 'MQTT'
            #protocol = 'MODBUS'
            protocols.append('MQTT')
            mqtt.update_many(
                { "Flow ID":item['value'] },
                {
                    "$set": { "Label": item['Label']},
                    #$currentDate: { lastModified: true }
                }
            )
            y = pd.concat([y,pd.DataFrame(list(mqtt.find({"Flow ID":item['value']})))]).reset_index(drop=True)
            ###mqtt_labels.append(item['Label'])
            ##y = pd.DataFrame(list(mqtt.find({"Flow ID":item['value']})))
            #############for j in range (0, 20):
            for j in range (0, 20):
                for i in range(0, len(y)):
            
                    convert_to_np(db["retrain_mqtt"],y.iloc[i,1:].to_dict())
                    convert_to_np(db["history_mqtt"],y.iloc[i,1:].to_dict())

        if y1.empty:
            pass
        else:
            mqtt_labels.append(item['Label'])    
        if z.empty:
            pass
        else:
            #protocol = 'NTP'
            protocols.append('NTP')
            ntp.update_many(
                { "Flow ID":item['value']},
                {
                    "$set": { "Label": item['Label']},
                    
                }
            )
            z = pd.concat([z,pd.DataFrame(list(ntp.find({"Flow ID":item['value']})))]).reset_index(drop=True)
            ########ntp_labels.append(item['Label'])
            ##z = pd.DataFrame(list(ntp.find({"Flow ID":item['value']})))
            for j in range (0, 20):
                for i in range(0, len(z)):
                    convert_to_np(db["retrain_ntp"],z.iloc[i,1:].to_dict())
                    convert_to_np(db["history_ntp"],z.iloc[i,1:].to_dict())

        if z1.empty:
            pass
        else:
            ntp_labels.append(item['Label'])    
        if a.empty:
            pass
        else:
            #protocol = 'RADIUS'
            protocols.append('RADIUS')
            radius.update_many(
                { "Flow ID":item['value'] },
                {
                    "$set": { "Label": item['Label']},
                    
                }
            )
            ##a = pd.DataFrame(list(radius.find({"Flow ID":item['value']})))
            a = pd.concat([a,pd.DataFrame(list(radius.find({"Flow ID":item['value']})))]).reset_index(drop=True)
            ######radius_labels.append(item['Label'])
            for j in range (0, 20):
                for i in range(0, len(a)):
                    convert_to_np(db["retrain_radius"],a.iloc[i,1:].to_dict())
                    convert_to_np(db["history_radius"],a.iloc[i,1:].to_dict())
        if a1.empty:
            pass
        else:
            radius_labels.append(item['Label']) 
        if b.empty:
            pass
        else:
            #protocol = 'BACNET'
            protocols.append('BACNET')
            radius.update_many(
                { "Flow ID":item['value'] },
                {
                    "$set": { "Label": item['Label']},
                    
                }
            )
            #####bacnet_labels.append(item['Label'])
            b = pd.concat([b,pd.DataFrame(list(bacnet.find({"Flow ID":item['value']})))]).reset_index(drop=True)
            ##b = pd.DataFrame(list(bacnet.find({"Flow ID":item['value']})))
            for j in range (0, 20):
                for i in range(0, len(b)):
                    convert_to_np(db["retrain_bacnet"],a.iloc[i,1:].to_dict())
                    convert_to_np(db["history_bacnet"],a.iloc[i,1:].to_dict())
        if b1.empty:
            pass
        else:
            bacnet_labels.append(item['Label']) 
        x1 = pd.DataFrame()
        y1 = pd.DataFrame()
        z1 = pd.DataFrame()
        a1 = pd.DataFrame()
        b1 = pd.DataFrame()   
        #In this phase, it is restored all the attacks of system in special databases like
        #mqtt_attacks etc. We have 4 cases in which we refferd in next comments.
    list_set_mqtt = set(mqtt_labels)
    # convert the set to the list
    mqtt_labels = (list(list_set_mqtt))

    list_set_modbus = set(modbus_labels)
    # convert the set to the list
    modbus_labels = (list(list_set_modbus))


    list_set_ntp = set(ntp_labels)
    # convert the set to the list
    ntp_labels = (list(list_set_ntp))

    list_set_radius = set(radius_labels)
    # convert the set to the list
    radius_labels = (list(list_set_radius))

    list_set_bacnet = set(bacnet_labels)
    # convert the set to the list
    bacnet_labels = (list(list_set_bacnet))
    
    list_set = set(protocols)
    # convert the set to the list
    protocols = (list(list_set))
    print(protocols)
    #print(x)
    print(y)
    #print(x["Label"])
    print(y["Label"])
    mqtt_attacks = db['mqtt_attacks']
    modbus_attacks = db['modbus_attacks']
    ntp_attacks = db['ntp_attacks']
    radius_attacks = db['radius_attacks']
    bacnet_attacks = db['bacnet_attacks']
    #For each protocol, it is took place the corresponding process
    for protocol in protocols:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        if protocol == 'MQTT':
            cnt=0
            attackLabelsToInt = {
                    "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
            for attack in mqtt_labels:
                   #1st: case mqtt_attacks is in mongodb collections. This means that
                # system works at least one time
                if "mqtt_attacks" in db.list_collection_names():
                    
                    #if the attack is in list of attacks means that it is not needs to retrain
                    #the system from scratch. It is updated the deep learning model with new data
                    #but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                    if attack in  list(list(list(mqtt_attacks.find())[0].items())[1][1]):
                            pass
                            ##num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                            ##attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                            ##retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    
                    
                    
                    #In case that we have a new attack:
                    #1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                    #2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                    #3. drop the old collection in mongodb that represents the old attack list
                    #4. Restore the new attack list
                    #5. Get the new number of classes
                    #Train all models from scratch with TrainModels module. The logits of deep learning 
                    #model are changed so it is obligable to be made this process.
                    
                    
                    else:

                            attack_dict = list(mqtt_attacks.find()[0].items())[1][1]
                            attack_dict['{}'.format(attack)] = list(list(mqtt_attacks.find()[0].items())[1][1].values())[-1]+1
                            print('**********************')
                            print('attack dict with db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt) 
                            db.mqtt_attacks.drop()
                            mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                            num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                            cnt+=1
                            ##TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                           
                            
                            
                else:
                    
                    #3rd case: In this case
                    #         a. The system not run in past\
                    #         b. The attack belongs to initial attack list
                    #In this case it is retrained the initail models of system
                    if attack in list(list(attackLabelsToInt.values())[0].keys()):
                            pass
                            #attackLabelsToInt = {
                               #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                            #mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                            #num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    #4th case: In this case
                    #          a. the system not run in past
                    #          b. the attack is totally new and models are trained from scratch.        
                    else:
                            attack_dict = list(attackLabelsToInt.values())[0]
                            attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1])+1
                            print('**********************')
                            print('attack dict without db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt)
                            num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                            cnt+=1
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            if cnt == 0:
                print('This is 0')
                if "mqtt_attacks" in db.list_collection_names():
                     num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                     attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                     print('########################')
                     print(attackLabelsToInt)
                     print('RETRAIN MODULE')
                     retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                else:
                    attackLabelsToInt = {
                    "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                    mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                    print('RETRAIN MODULE')
                    retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
            else:
                if "mqtt_attacks" in db.list_collection_names():
                    #pass
                    print('There is an mwtt')
                    attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                    print('TRAIN MODULE')
                    TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                else:
                    print('TRAIN MODULE')
                    #pass
                    TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            #if "mqtt_attacks" in db.list_collection_names():
                #attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
            print('**********************')
            #db.modbus_attacks.drop()
            print('MQTT_labels')
            print(num_classes)
            #print(attackLabelsToInt)
            print(attackLabelsToInt)
        if protocol == 'MODBUS':
            cnt=0
            attackLabelsToInt = {
                                "Label": {'Normal':0, 'UID brute force':1, 'Enumeration Function':2, 'Fuzzing Read Holding Registers':3}}
            for attack in modbus_labels:
                   #!st: case mqtt_attacks is in mongodb collections. This means that
                # system works at least one time
                if "modbus_attacks" in db.list_collection_names():
                    
                    #if the attack is in list of attacks means that it is not needs to retrain
                    #the system from scratch. It is updated the deep learning model with new data
                    #but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                    if attack in  list(list(list(modbus_attacks.find())[0].items())[1][1]):
                            pass
                            ##num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                            ##attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                            ##retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    
                    
                    
                    #In case that we have a new attack:
                    #1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                    #2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                    #3. drop the old collection in mongodb that represents the old attack list
                    #4. Restore the new attack list
                    #5. Get the new number of classes
                    #Train all models from scratch with TrainModels module. The logits of deep learning 
                    #model are changed so it is obligable to be made this process.
                    
                    
                    else:

                            attack_dict = list(modbus_attacks.find()[0].items())[1][1]
                            attack_dict['{}'.format(attack)] = list(list(modbus_attacks.find()[0].items())[1][1].values())[-1]+1
                            print('********************')
                            print('attack dict with db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt)
                            db.modbus_attacks.drop()
                            mycol = db["modbus_attacks"].insert_one(attackLabelsToInt)
                            num_classes = len(list(list(modbus_attacks.find()[0].items())[1][1].values()))
                            cnt+=1
                            print(cnt)
                            ##TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                            
                            
                            
                else:
                    
                    #3rd case: In this case
                    #         a. The system not run in past\
                    #         b. The attack belongs to initial attack list
                    #In this case it is retrained the initail models of system
                    if attack in list(list(attackLabelsToInt.values())[0].keys()):
                            pass
                            #attackLabelsToInt = {
                               #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                            #mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                            #num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    #4th case: In this case
                    #          a. the system not run in past
                    #          b. the attack is totally new and models are trained from scratch.        
                    else:
                            attack_dict = list(attackLabelsToInt.values())[0]
                            attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1])+1
                            print('*********************')
                            print('attack dict without db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt)
                            num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            mycol = db["modbus_attacks"].insert_one(attackLabelsToInt)
                            cnt+=1
                            print(cnt)
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            if cnt == 0:
                print('This is 0')
                if "modbus_attacks" in db.list_collection_names():
                     num_classes = len(list(list(modbus_attacks.find()[0].items())[1][1].values()))
                     attackLabelsToInt = {"Label": list(list(modbus_attacks.find())[0].items())[1][1]}
                     print('########################')
                     print(attackLabelsToInt)
                     print("RETRAIN MODULE")
                     retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                else:
                     #print('MODBUS network flow classification test')
                    attackLabelsToInt = {
                                "Label": {'Normal':0, 'UID brute force':1, 'Enumeration Function':2, 'Fuzzing Read Holding Registers':3}}
                    #attackLabelsToInt = {
                    #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                    mycol = db["modbus_attacks"].insert_one(attackLabelsToInt)
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    attackLabelsToInt = {"Label": list(list(modbus_attacks.find())[0].items())[1][1]}
                    print("RETRAIN MODULE")
                    retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
            else:
                if "modbus_attacks" in db.list_collection_names():
                    print('There is an modbus')
                    attackLabelsToInt = {"Label": list(list(modbus_attacks.find())[0].items())[1][1]}
                    print("TRAIN MODULE")
                    #pass
                    TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                else:
                    print("TRAIN MODULE")
                    #pass
                    TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            #if "modbus_attacks" in db.list_collection_names():
                #attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
            print('**********************')
            #db.mqtt_attacks.drop()
            print('MODBUS_labels')
            print(num_classes)
            #print(attackLabelsToInt)
            #print(modbus_labels)
            print(attackLabelsToInt)
        if protocol == 'BACNET':
            cnt=0
            attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
            
            for attack in bacnet_labels:
                   #!st: case mqtt_attacks is in mongodb collections. This means that
                # system works at least one time
                if "bacnet_attacks" in db.list_collection_names():
                    
                    #if the attack is in list of attacks means that it is not needs to retrain
                    #the system from scratch. It is updated the deep learning model with new data
                    #but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                    if attack in  list(list(list(bacnet_attacks.find())[0].items())[1][1]):
                            pass
                            ##num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                            ##attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                            ##retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    
                    
                    
                    #In case that we have a new attack:
                    #1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                    #2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                    #3. drop the old collection in mongodb that represents the old attack list
                    #4. Restore the new attack list
                    #5. Get the new number of classes
                    #Train all models from scratch with TrainModels module. The logits of deep learning 
                    #model are changed so it is obligable to be made this process.
                    
                    
                    else:

                            attack_dict = list(bacnet_attacks.find()[0].items())[1][1]
                            attack_dict['{}'.format(attack)] = list(list(bacnet_attacks.find()[0].items())[1][1].values())[-1]+1
                            print('********************')
                            print('attack dict with db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt)
                            db.bacnet_attacks.drop()
                            mycol = db["bacnet_attacks"].insert_one(attackLabelsToInt)
                            num_classes = len(list(list(bacnet_attacks.find()[0].items())[1][1].values()))
                            cnt+=1
                            print(cnt)
                            ##TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                            
                            
                            
                else:
                    
                    #3rd case: In this case
                    #         a. The system not run in past\
                    #         b. The attack belongs to initial attack list
                    #In this case it is retrained the initail models of system
                    if attack in list(list(attackLabelsToInt.values())[0].keys()):
                            pass
                            #attackLabelsToInt = {
                               #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                            #mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                            #num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    #4th case: In this case
                    #          a. the system not run in past
                    #          b. the attack is totally new and models are trained from scratch.        
                    else:
                            attack_dict = list(attackLabelsToInt.values())[0]
                            attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1])+1
                            print('*********************')
                            print('attack dict without db') 
                            print(attack_dict)
                            attackLabelsToInt = {"Label":attack_dict}
                            print(attackLabelsToInt)
                            num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            mycol = db["bacnet_attacks"].insert_one(attackLabelsToInt)
                            cnt+=1
                            print(cnt)
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            if cnt == 0:
                print('This is 0')
                if "bacnet_attacks" in db.list_collection_names():
                     num_classes = len(list(list(bacnet_attacks.find()[0].items())[1][1].values()))
                     attackLabelsToInt = {"Label": list(list(bacnet_attacks.find())[0].items())[1][1]}
                     print('########################')
                     print(attackLabelsToInt)
                     print("RETRAIN MODULE")
                     #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                else:
                     #print('MODBUS network flow classification test')
                    attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
                    #attackLabelsToInt = {
                    #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                            

                    mycol = db["bacnet_attacks"].insert_one(attackLabelsToInt)
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    attackLabelsToInt = {"Label": list(list(bacnet_attacks.find())[0].items())[1][1]}
                    print("RETRAIN MODULE")
                    #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
            else:
                if "bacnet_attacks" in db.list_collection_names():
                    print('There is an modbus')
                    attackLabelsToInt = {"Label": list(list(bacnet_attacks.find())[0].items())[1][1]}
                    print("TRAIN MODULE")
                    #pass
                    #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                else:
                    print("TRAIN MODULE")
                    #pass
                    #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
            #if "modbus_attacks" in db.list_collection_names():
                #attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
            print('**********************')
            #db.mqtt_attacks.drop()
            print('MODBUS_labels')
            print(num_classes)
            #print(attackLabelsToInt)
            #print(modbus_labels)
            print(attackLabelsToInt)          
        if protocol == 'NTP':
                    cnt=0
                    attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
                    
                    for attack in ntp_labels:
                        #!st: case mqtt_attacks is in mongodb collections. This means that
                        # system works at least one time
                        if "ntp_attacks" in db.list_collection_names():
                            
                            #if the attack is in list of attacks means that it is not needs to retrain
                            #the system from scratch. It is updated the deep learning model with new data
                            #but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                            if attack in  list(list(list(ntp_attacks.find())[0].items())[1][1]):
                                    pass
                                    ##num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                                    ##attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                                    ##retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                            
                            
                            
                            #In case that we have a new attack:
                            #1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                            #2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                            #3. drop the old collection in mongodb that represents the old attack list
                            #4. Restore the new attack list
                            #5. Get the new number of classes
                            #Train all models from scratch with TrainModels module. The logits of deep learning 
                            #model are changed so it is obligable to be made this process.
                            
                            
                            else:

                                    attack_dict = list(ntp_attacks.find()[0].items())[1][1]
                                    attack_dict['{}'.format(attack)] = list(list(ntp_attacks.find()[0].items())[1][1].values())[-1]+1
                                    print('********************')
                                    print('attack dict with db') 
                                    print(attack_dict)
                                    attackLabelsToInt = {"Label":attack_dict}
                                    print(attackLabelsToInt)
                                    db.ntp_attacks.drop()
                                    mycol = db["ntp_attacks"].insert_one(attackLabelsToInt)
                                    num_classes = len(list(list(ntp_attacks.find()[0].items())[1][1].values()))
                                    cnt+=1
                                    print(cnt)
                                    ##TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                                    
                                    
                                    
                        else:
                            
                            #3rd case: In this case
                            #         a. The system not run in past\
                            #         b. The attack belongs to initial attack list
                            #In this case it is retrained the initail models of system
                            if attack in list(list(attackLabelsToInt.values())[0].keys()):
                                    pass
                                    #attackLabelsToInt = {
                                    #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    

                                    #mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                                    #num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                                    #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                            #4th case: In this case
                            #          a. the system not run in past
                            #          b. the attack is totally new and models are trained from scratch.        
                            else:
                                    attack_dict = list(attackLabelsToInt.values())[0]
                                    attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1])+1
                                    print('*********************')
                                    print('attack dict without db') 
                                    print(attack_dict)
                                    attackLabelsToInt = {"Label":attack_dict}
                                    print(attackLabelsToInt)
                                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                                    mycol = db["ntp_attacks"].insert_one(attackLabelsToInt)
                                    cnt+=1
                                    print(cnt)
                                    #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
                    if cnt == 0:
                        print('This is 0')
                        if "ntp_attacks" in db.list_collection_names():
                            num_classes = len(list(list(ntp_attacks.find()[0].items())[1][1].values()))
                            attackLabelsToInt = {"Label": list(list(ntp_attacks.find())[0].items())[1][1]}
                            print('########################')
                            print(attackLabelsToInt)
                            print("RETRAIN MODULE")
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                        else:
                            #print('MODBUS network flow classification test')
                            attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}
                            #attackLabelsToInt = {
                            #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    

                            mycol = db["ntp_attacks"].insert_one(attackLabelsToInt)
                            num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            attackLabelsToInt = {"Label": list(list(ntp_attacks.find())[0].items())[1][1]}
                            print("RETRAIN MODULE")
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    else:
                        if "ntp_attacks" in db.list_collection_names():
                            print('There is an modbus')
                            attackLabelsToInt = {"Label": list(list(ntp_attacks.find())[0].items())[1][1]}
                            print("TRAIN MODULE")
                            #pass
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                        else:
                            print("TRAIN MODULE")
                            #pass
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
                    #if "modbus_attacks" in db.list_collection_names():
                        #attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                    print('**********************')
                    #db.mqtt_attacks.drop()
                    print('MODBUS_labels')
                    print(num_classes)
                    #print(attackLabelsToInt)
                    #print(modbus_labels)
                    print(attackLabelsToInt)          
        if protocol == 'RADIUS':
                    cnt=0
                    attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
                    
                    for attack in radius_labels:
                        #!st: case mqtt_attacks is in mongodb collections. This means that
                        # system works at least one time
                        if "radius_attacks" in db.list_collection_names():
                            
                            #if the attack is in list of attacks means that it is not needs to retrain
                            #the system from scratch. It is updated the deep learning model with new data
                            #but machine learning models are trained from scratch. In line 263 it is called the retrain module.
                            if attack in  list(list(list(radius_attacks.find())[0].items())[1][1]):
                                    pass
                                    ##num_classes = len(list(list(mqtt_attacks.find()[0].items())[1][1].values()))
                                    ##attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                                    ##retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                            
                            
                            
                            #In case that we have a new attack:
                            #1. make a dictionar of attacks from last mqtt_attacks like {Normal:0, old_attack:1}
                            #2. we create a new dictionary like: {Normal:0, old_attack:1, new_attack:2}
                            #3. drop the old collection in mongodb that represents the old attack list
                            #4. Restore the new attack list
                            #5. Get the new number of classes
                            #Train all models from scratch with TrainModels module. The logits of deep learning 
                            #model are changed so it is obligable to be made this process.
                            
                            
                            else:

                                    attack_dict = list(radius_attacks.find()[0].items())[1][1]
                                    attack_dict['{}'.format(attack)] = list(list(radius_attacks.find()[0].items())[1][1].values())[-1]+1
                                    print('********************')
                                    print('attack dict with db') 
                                    print(attack_dict)
                                    attackLabelsToInt = {"Label":attack_dict}
                                    print(attackLabelsToInt)
                                    db.radius_attacks.drop()
                                    mycol = db["radius_attacks"].insert_one(attackLabelsToInt)
                                    num_classes = len(list(list(radius_attacks.find()[0].items())[1][1].values()))
                                    cnt+=1
                                    print(cnt)
                                    ##TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                                    
                                    
                                    
                        else:
                            
                            #3rd case: In this case
                            #         a. The system not run in past\
                            #         b. The attack belongs to initial attack list
                            #In this case it is retrained the initail models of system
                            if attack in list(list(attackLabelsToInt.values())[0].keys()):
                                    pass
                                    #attackLabelsToInt = {
                                    #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    

                                    #mycol = db["mqtt_attacks"].insert_one(attackLabelsToInt)
                                    #num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                                    #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                            #4th case: In this case
                            #          a. the system not run in past
                            #          b. the attack is totally new and models are trained from scratch.        
                            else:
                                    attack_dict = list(attackLabelsToInt.values())[0]
                                    attack_dict['{}'.format(attack)] = (list(list(attackLabelsToInt.values())[0].values())[-1])+1
                                    print('*********************')
                                    print('attack dict without db') 
                                    print(attack_dict)
                                    attackLabelsToInt = {"Label":attack_dict}
                                    print(attackLabelsToInt)
                                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                                    mycol = db["radius_attacks"].insert_one(attackLabelsToInt)
                                    cnt+=1
                                    print(cnt)
                                    #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
                    if cnt == 0:
                        print('This is 0')
                        if "ntp_attacks" in db.list_collection_names():
                            num_classes = len(list(list(radius_attacks.find()[0].items())[1][1].values()))
                            attackLabelsToInt = {"Label": list(list(radius_attacks.find())[0].items())[1][1]}
                            print('########################')
                            print(attackLabelsToInt)
                            print("RETRAIN MODULE")
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                        else:
                            #print('MODBUS network flow classification test')
                            attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
                            #attackLabelsToInt = {
                            #"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    

                            mycol = db["radius_attacks"].insert_one(attackLabelsToInt)
                            num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                            attackLabelsToInt = {"Label": list(list(radius_attacks.find())[0].items())[1][1]}
                            print("RETRAIN MODULE")
                            #retrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt)
                    else:
                        if "radius_attacks" in db.list_collection_names():
                            print('There is an modbus')
                            attackLabelsToInt = {"Label": list(list(radius_attacks.find())[0].items())[1][1]}
                            print("TRAIN MODULE")
                            #pass
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack = attack)
                        else:
                            print("TRAIN MODULE")
                            #pass
                            #TrainModels(protocol = protocol, attackLabelsToInt = attackLabelsToInt,  attack=attack)
                    #if "modbus_attacks" in db.list_collection_names():
                        #attackLabelsToInt = {"Label": list(list(mqtt_attacks.find())[0].items())[1][1]}
                    print('**********************')
                    #db.mqtt_attacks.drop()
                    print('MODBUS_labels')
                    print(num_classes)
                    #print(attackLabelsToInt)
                    #print(modbus_labels)
                    print(attackLabelsToInt)                
    
    
    return 'database is updated and training process has been started',200


def convert_to_np(collection,r):
    
    try:
        collection.insert(r)
    except pymongo.errors.InvalidDocument:
        # Python 2.7.10 on Windows and Pymongo are not forgiving
        # If you have foreign data types you have to convert them
        n = {}
        for k, v in r.items():
            
            if isinstance(v, np.int64):
                
                v = int(v)
                

            n[k] = v

        collection.insert(n)
def TrainModels(protocol, attackLabelsToInt,  attack):
    #As we make the same training process we hard-coded the request
    #for training in parallel process
    protocol = protocol
    
    modelTypes= ["SDAE", "KNN","LogReg", "Random-Forest"]
    #modelTypes= ["SDAE", "SVC"]
    #modelTypes= ["KNN"]
    
    crossVal = 2
    
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    start = time.time()
    request_body = request.get_json() 
    protocols = config ['Protocols']['protocols'].split(',')
    all_models = config['Model_types']['model_types'].split(',')
    protocol = protocol#request_body['protocol']
    #error handling in case that protocols and ModelType is not in predifined
    if protocol in protocols:
        pass
    else:
        return jsonify({"error": "Invalid Requested Protocol",}), 400
    
    
    if (modelTypes in all_models) or (set(modelTypes).issubset(set(all_models))):
        pass
    else:
        return jsonify({"error": "Invalid Requested Model",}), 400
    
    end = time.time()
    
    #if (end - start) < 10:
        #pass
    #else:
        #return jsonify({"error": "Request Timeout Error",}), 504
    #we send request to be initialized the parallel process
    ##train_task = Process(target=train, args=(create_data_init_train(protocol, modelTypes, attackLabelsToInt), protocol, modelTypes, crossVal,  attack))
    train_task = Process(target=train, args=(create_data(protocol, modelTypes, attackLabelsToInt), protocol, modelTypes, crossVal,  attack))
    
    train_task.start()

    return 'Model re-training  started'
def retrainModels(protocol, attackLabelsToInt):
    protocol = protocol
    modelTypes= ["SDAE", "KNN", "LogReg", "Random-Forest"]
    #modelTypes= ["SDAE","SVC"]
    #modelTypes= ["KNN"]
    
    crossVal = 2
    print('Starting retraining process')
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    start = time.time()
    request_body = request.get_json() 
    protocols = config ['Protocols']['protocols'].split(',')
    all_models = config['Model_types']['model_types'].split(',')
    protocol = protocol#request_body['protocol']
    
    if protocol in protocols:
        pass
    else:
        return jsonify({"error": "Invalid Requested Protocol",}), 400
    
    
    if (modelTypes in all_models) or (set(modelTypes).issubset(set(all_models))):
        pass
    else:
        return jsonify({"error": "Invalid Requested Model",}), 400
    end = time.time()
    
    #if (end - start) < 10:
        #pass
    #else:
        #return jsonify({"error": "Request Timeout Error",}), 504
    
    #We make the same process as TrainModels for retraining module
    train_task = Process(target=Retrain, args=(protocol, modelTypes, crossVal, attackLabelsToInt))
    
    train_task.start()

    return 'Model re-training  started'
def create_data(protocol, model, attackLabelsToInt):
    #It is concated the data points from csvs and the data points from historic data 
    #of historic collection
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    modbus_retr = db["history_modbus"]
    mqtt_retr = db["history_mqtt"]
    backnet_retr = db['history_bacnet']
    ntp_retr = db['history_ntp']
    radius_retr = db['history_radius']
    print('CREATE DATA ALTI')
    print(attackLabelsToInt)
    #if model == "SDAE" or model == "DNN":
    if protocol == 'MQTT':
            
            #initial csv is read
            csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
            #1. we take into account the historical data from user interactions from the 
            #past and now.
            mqtt = db["history_mqtt"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            print(attackLabelsToInt)
            for item in mqtt.find():

                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
            #historic data and csv are concatenated in order to be formed a nes dataframe    
            len_attacks = int(len(new_dataframe)/20)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:] 
            csv.replace(attackLabelsToInt, inplace = True)  
    elif protocol == 'BACNET':
            #init_csv = pd.read_csv("datasets/bacnet_dataset.csv")
            #csv = pd.DataFrame(list(backnet_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)


            csv = pd.read_csv("datasets/bacnet_dataset.csv")
            
            bacnet = db["history_bacnet"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            print(attackLabelsToInt)
            for item in bacnet.find():
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = int(len(new_dataframe)/20)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
            csv.replace(attackLabelsToInt, inplace = True)
    elif protocol == 'MODBUS':
            #init_csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
            #csv = pd.DataFrame(list(modbus_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)
            csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
            
            modbus = db["history_modbus"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            print(attackLabelsToInt)
            for item in modbus.find():
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = int(len(new_dataframe)/20)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]

            csv.replace(attackLabelsToInt, inplace = True)
            
    elif protocol == 'NTP':
            #init_csv = pd.read_csv("datasets/ntp_dataset_1.csv")
            #csv = pd.DataFrame(list(ntp_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)


            csv = pd.read_csv("datasets/ntp_dataset_1.csv")
            
            ntp = db["history_ntp"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            print(attackLabelsToInt)
            for item in ntp.find():
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = int(len(new_dataframe)/20)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
            csv.replace(attackLabelsToInt, inplace = True)
    
    elif protocol == 'RADIUS':
            #init_csv = pd.read_csv("datasets/radius_dataset.csv")
            #csv = pd.DataFrame(list(radius_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]

            #len_attacks = len(csv)


            csv = pd.read_csv("datasets/radius_dataset.csv")
            
            radius = db["history_radius"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            print(attackLabelsToInt)
            for item in radius.find():
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            #len_attacks = int(len(new_dataframe)/20)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
            csv.replace(attackLabelsToInt, inplace = True)
    return csv#, len_attacks
def train(data, protocol, models, cv,  attack):
    # register SPARK backend
    register_spark()

    start_time = time.time()
    
    results = Parallel(backend='spark', n_jobs=len(models))(
        delayed(parallelTrain)(data, protocol, model, cv, attack) for model in models)

    print(results)

    best_f1 = 0
    best_model = 'some tuples are None'
    for tup in results:
        if tup!=None:
            if tup[1]!=np.nan:    
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



def parallelTrain(csv, protocol, model_type, cv,  attack):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    data = Data(csv, protocol, model_type)
    #It is called the training module
    x_train, x_test, y_train, y_test = data.train_test_data()
    #if model_type != 'SDAE':
        #pass
    #else:
    train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv,db, attack)
    #(self, x_train, y_train, x_test, y_test, data, protocol, model_type, cv)       
    
    report = train.classify()
    print('F1 score for ', model_type , ': ' ,  report.get('weighted avg').get('f1-score'))
    f1 = report.get('macro avg').get('f1-score')
    return model_type, f1
r'''
def create_data_init_train(protocol, model, attackLabelsToInt):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    modbus_retr = db["history_modbus"]
    mqtt_retr = db["history_mqtt"]
    backnet_retr = db['history_bacnet']
    ntp_retr = db['history_ntp']
    radius_retr = db['retrain_radius']
    
    if protocol == 'MQTT':
           
            
            csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
            
            
            mqtt = db["history_mqtt"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            
            for item in mqtt.find():
                
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                dataframe.replace(attackLabelsToInt, inplace = True)
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = len(new_dataframe)/20
            
            csv.replace(attackLabelsToInt, inplace = True)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:] 
               
    elif protocol == 'BACNET':
            #init_csv = pd.read_csv("datasets/bacnet_dataset.csv")
            #csv = pd.DataFrame(list(backnet_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)


            csv = pd.read_csv("datasets/bacnet_dataset.csv")
            
            
            bacnet = db["history_bacnet"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            
            for item in bacnet.find():
                
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                dataframe.replace(attackLabelsToInt, inplace = True)
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = len(new_dataframe)/20
            
            csv.replace(attackLabelsToInt, inplace = True)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
            

    elif protocol == 'MODBUS':
            #init_csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
            #csv = pd.DataFrame(list(modbus_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)


            csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
            
            
            modbus = db["history_modbus"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            
            for item in modbus.find():
                
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                dataframe.replace(attackLabelsToInt, inplace = True)
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = len(new_dataframe)/20
            
            csv.replace(attackLabelsToInt, inplace = True)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
    elif protocol == 'NTP':
            #init_csv = pd.read_csv("datasets/ntp_dataset_1.csv")
            #csv = pd.DataFrame(list(ntp_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]
            #len_attacks = len(csv)

            csv = pd.read_csv("datasets/ntp_dataset_1.csv")
            
            
            ntp = db["history_ntp"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            
            for item in ntp.find():
                
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                dataframe.replace(attackLabelsToInt, inplace = True)
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = len(new_dataframe)/20
            
            csv.replace(attackLabelsToInt, inplace = True)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
    elif protocol == 'RADIUS':
            #init_csv = pd.read_csv("datasets/radius_dataset.csv")
            #csv = pd.DataFrame(list(radius_retr.find()))
            
            #csv.replace(attackLabelsToInt, inplace = True)
            #columns  = list(set(csv.columns).intersection(init_csv.columns))
            #csv = csv[columns]

            #len_attacks = len(csv)

            csv = pd.read_csv("datasets/radius_dataset.csv")
            
            
            radius = db["history_radius"]
            
            new_dataframe = pd.DataFrame()
            counter = 0
            
            attackLabelsToInt = {'Label':list(attackLabelsToInt.values())[0]}
            
            for item in radius.find():
                
                dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
                dataframe.replace(attackLabelsToInt, inplace = True)
                columns  = list(set(dataframe.columns).intersection(csv.columns))
                dataframe = dataframe[columns]
                new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
                
            len_attacks = len(new_dataframe)/20
            
            csv.replace(attackLabelsToInt, inplace = True)
            csv = pd.concat([csv,new_dataframe]).reset_index(drop=True)#.iloc[-counter:]
    
    
    return csv

'''


def Retrain(protocol, models, cv, attackLabelsToInt):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # register SPARK backend
    register_spark()
    
    start_time = time.time()
    
    results = Parallel(backend='spark', n_jobs=len(models))(
        delayed(parallelRetrain)(create_data(protocol, model,attackLabelsToInt), protocol, model, cv, attackLabelsToInt) for model in models)
        
    
    best_f1 = 0
    print('best f1 is')
    print(best_f1)
    print(results)
    for tup in results:
        if tup[1] > best_f1 and tup!= np.nan:
            best_f1 = tup[1]
            best_model: object = tup[0]
    print('Best model: ', best_model, ' with best value: ', best_f1)
    print('Execution time of model %s seconds' % (time.time() - start_time))
    if best_model == 'SDAE':
        print('best  models is DL')
        
        model = tf.keras.models.load_model("models/" + protocol + "_" + best_model + "_model.h5")
        model.save('best_models/' + protocol + "_" + best_model + '_model.h5')
        
    else:
        print('best  models is ML')
        model = pickle.load(open('models/' + protocol + "_" + best_model + '_model.pkl', 'rb'))
        with open('best_models/' + protocol + "_" + best_model + '_model.pkl', 'wb') as fid:
            pickle.dump(model, fid)
    #'''
    print('Execution time: %s seconds' % (time.time() - start_time))
    
    if "retrain_mqtt" in db.list_collection_names():
                db.retrain_mqtt.drop()
                
                print('RETRAIN DB IS APPEARED')
     
def parallelRetrain(csv, protocol, model_type, cv, attackLabelsToInt):
    
    
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient["labelled_netflows_db"]
    if protocol == 'MQTT':
        csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
    elif protocol == 'BACNET':
        csv = pd.read_csv("datasets/bacnet_dataset.csv")
    elif protocol == "MODBUS":
        csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
    elif protocol == 'NTP':
        csv = pd.read_csv("datasets/ntp_dataset_1.csv")
    elif protocol == 'RADIUS':
        csv = pd.read_csv("datasets/radius_dataset.csv")
    my_csv = csv.copy()#pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
    #In this function we have some special steps
    #1. we take into account the historical data from user interactions from the 
    #past and now.
    #2. We get some random samples from initail csv as to have the same number of records
    #for each attack
    #3. Concatenate the historic interactions in pandas dataframe format with random samples 
    #of initial csvs.
    #4. We get the columns that is the same from initial csvs and the collection historic data.
    #5. We have 76 features
    if model_type == "SDAE" or model_type == "DNN":
        
            #len_attacks = create_data(protocol, model_type, attackLabelsToInt)#[1]
            
            
            attack_list = []
            
            for k in attackLabelsToInt['Label']:
                
                attack_list.append(k)
            
            counter = 0
            limit = 64#-len_attacks
            print('inside parallel retrain')
            final_df = pd.DataFrame(columns = csv.columns)
            print(limit)
            while final_df.empty:
                for item in attack_list:
                    #2. We get some random samples from initail csv as to have the same number of records
                    #for each attack
                    med_df = (csv[csv['Label']==item])
                    if len(med_df)> int(limit/len(attack_list)):
                        new_df = med_df.sample(n= int(limit/len(attack_list)))
                    else:
                        new_df = med_df.copy()
                    #4. We get the columns that is the same from initial csvs and the collection historic data.
                    columns  = list(set(final_df.columns).intersection(csv.columns))
                    new_df = new_df[columns]
                    
                    final_df = pd.concat([final_df, new_df])
                    print('final_df_is')
                    print(final_df)
            print('out of first while loop')
                    
            list_indices = final_df.index.values.tolist()
            #We get some other random samples to have at least 64 data points.
            #Otherwise we have bug in retraining process
            other_df = pd.DataFrame()
            while other_df.empty:
                if len(final_df) < 64:
                        limit = 64- len(final_df)
                        other_df = csv.sample(n= limit)
                        columns  = list(set(other_df.columns).intersection(final_df.columns))
                        other_df = other_df[columns]#.reset_index(drop=True)
                        other_df = other_df.dropna()
                else:
                 	    break
            print('out of second loop')

            other_list_indices = other_df.index.values.tolist()
            final_list_indices = list_indices + other_list_indices
            final_df =  pd.concat([final_df, other_df])
            print('THIS IS DF FROM CSVS')
            print(final_df)
            ##1. we take into account the historical data from user interactions from the 
            #past and now.
            #x = pd.DataFrame(list(modbus.find({"Flow ID":netflow_id})))
            train_db = pd.DataFrame(list(db["history_{}".format(protocol.lower())].find()))
            print(train_db)
            #train_db = create_data(protocol, model_type, attackLabelsToInt)[0]#.iloc[-len_attacks:, :]
            final_columns  = list(set(train_db.columns).intersection(final_df.columns))
            
            train_db = train_db[final_columns]
            print('THIS IS DB FROM COLLECTIONS')
            print(train_db)
            #4. We get the columns that is the same from initial csvs and the collection historic data.
            final_df = pd.concat([final_df, train_db])
            final_df = final_df.reset_index(drop=True)
            if "mqtt_attacks" in db.list_collection_names():
                 attackLabelsToInt = list(attackLabelsToInt.values())[0]#pass
            
            final_df.replace(attackLabelsToInt, inplace = True)
            data = Data(final_df, protocol, model_type)
            x_train, x_test, y_train, y_test = data.train_test_data()
            print('x retrain is:', x_train)
            print('x retrain test is:', x_test)
            print('y retrain is:', y_train)
            print('y retrain test is:', y_test)
            print('shapes of retraining data', x_train.shape,y_train.shape,x_test.shape,y_train.shape)
            x_nan = x_train[np.isnan(x_train)]
            
            nan_array = np.isnan(x_train)
            print(nan_array)
            not_nan_array = ~ nan_array
            print(not_nan_array)
            x = x_train[not_nan_array]
            x_nan = x_train[~np.isnan(x_train)]  
            train = Training_DL(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv,db)
            report = train.classify(db)
            f1 = report.get('macro avg').get('f1-score')
            return model_type, f1
            
    else:
        #In this case we get only the concatenated data from csv and database collection in order
        #to be make train from scratch
            
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        db = myclient["labelled_netflows_db"]
        csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
        #my_csv = csv.copy()#pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
        #len_attacks = create_data(protocol, model_type, attackLabelsToInt)[1]         
        attack_list = []
        for k in attackLabelsToInt['Label']:
                    
                    attack_list.append(k)
        r'''
        counter = 0
        limit = 64-len_attacks
        final_df = pd.DataFrame(columns = csv.columns)
        
        while final_df.empty:
                    for item in attack_list:
                        med_df = (csv[csv['Label']==item])
                        if len(med_df)> int(limit/len(attack_list)):
                            new_df = med_df.sample(n= int(limit/len(attack_list)))
                        else:
                            new_df = med_df.copy()
                        
                        columns  = list(set(final_df.columns).intersection(csv.columns))
                        new_df = new_df[columns]
                        final_df = pd.concat([final_df, new_df])
                        
    
                
        
            
        list_indices = final_df.index.values.tolist()
        ######
        other_df = pd.DataFrame()
        while other_df.empty:
                    if len(final_df) < 64:
                            limit = 64- len(final_df)
                            other_df = csv.sample(n= limit)
                            columns  = list(set(other_df.columns).intersection(final_df.columns))
                            other_df = other_df[columns]#.reset_index(drop=True)
                            other_df = other_df.dropna()
                        

        other_list_indices = other_df.index.values.tolist()
        final_list_indices = list_indices + other_list_indices
        final_df =  pd.concat([final_df, other_df])
        print(final_df)
        train_db = create_data(protocol, model_type, attackLabelsToInt)[0]#.iloc[-len_attacks:, :]
        print('**************************')
        print('THIS IS TRAIN DB')
        print(train_db)
        final_columns  = list(set(train_db.columns).intersection(final_df.columns))
        train_db = train_db[final_columns]
        print(train_db)
        final_df = pd.concat([final_df, train_db])
        final_df = final_df.reset_index(drop=True)
        print('**************************')
        print('THIS IS FINAL DF')
        print(final_df)
        '''
        #In this case we get only the concatenated data from csv and database collection in order
        #to be make train from scratch
        train_db = create_data(protocol, model_type, attackLabelsToInt)#[0]#.iloc[-len_attacks:, :]
        print('**************************')
        print('THIS IS TRAIN DB')
        print(train_db)
        data = Data(train_db, protocol, model_type)
        x_train, x_test, y_train, y_test = data.train_test_data()
        #data = Data(final_df, protocol, model_type)
        #x_train, x_test, y_train, y_test = data.train_test_data()
        train = Training_ML(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv, db)
        #train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv)
            
        report = train.classify(db)
        print('F1 score for ', model_type , ': ' ,  report.get('weighted avg').get('f1-score'))
        f1 = report.get('macro avg').get('f1-score')
        return model_type, f1
        #''' 
app.run(host='localhost')
