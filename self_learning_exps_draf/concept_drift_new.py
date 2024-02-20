import pymongo
import time
import configparser
import pandas as pd
import scipy.spatial.distance as distance
import math 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json
from os import lchown
from flask import Flask, request, jsonify,json, render_template
import pymongo
import time
import configparser
import pandas as pd
from multiprocessing import Process
from joblibspark import register_spark
from retraining_exp import Training_ML, Training_DL
from training_exp import Training
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
import json
import os
#import retrain_models
#tf.compat.v1.disable_eager_execution()
config = configparser.ConfigParser()
config.read('config.ini')
        

class concept_drift:
    def data_processing(self, file):
        f = open(file)
        data = json.load(f)
        #print(data)
        resp_protocols = ['Mqtt','Modbus','Bacnet','Ntp','Radius']
        final_responses_mqtt = []
        final_responses_modbus = []
        final_responses_bacnet = []
        final_responses_ntp = []
        final_responses_radius = []
        #cnt=0
        for item in data:
            #print(item)
            if item['str_protocol'] in resp_protocols:
                #print(item['str_protocol'])
                
                if item['str_protocol'] == 'Mqtt':
                    final_responses_mqtt.append(item)
                if item['str_protocol'] == 'Modbus':
                    final_responses_modbus.append(item)
                if item['str_protocol'] == 'Bacnet':
                    final_responses_bacnet.append(item)
                if item['str_protocol'] == 'Ntp':
                    final_responses_ntp.append(item)
                if item['str_protocol'] == 'Radius':
                    final_responses_radius.append(item)
        if os.path.exists('labelled_mqtt.json'):
            with open('labelled_mqtt.json', "w") as file:
                
                data = final_responses_mqtt
                for item in data:
                    if item['Label'] == 'No Label':
                        item['Label'] = 'Normal'
                
                file.seek(0)
                json.dump(data, file)
                
                            
        else:
            if len(final_responses_mqtt) == 0:
                pass
            else:
                with open('labelled_mqtt.json', "w") as file:
                    for item in final_responses_mqtt:
                        if item['Label'] == 'No Label':
                            item['Label'] = 'Normal'
                    json.dump(final_responses_mqtt, file)

        if os.path.exists('labelled_modbus.json'):
            with open('labelled_modbus.json', "w") as file:
                #data = json.load(file)
                #print(type(data))
                #data = data[:-1]+','+ ' '.join([str(elem) for elem in final_responses_modbus][1:])
                data = final_responses_modbus
                for item in data:
                    if item['Label'] == 'No Label':
                        item['Label'] = 'Normal'
                #print(data)
                file.seek(0)
                json.dump(data, file)
                            
        else:
            if len(final_responses_modbus) == 0:
                pass
            else:
                with open('labelled_modbus.json', "w") as file:
                    json.dump(final_responses_modbus, file)

        if os.path.exists('labelled_bacnet.json'):
            with open('labelled_bacnet.json', "w") as file:
                #data = json.load(file)
                #print(type(data))
                #data = data[:-1]+','+ ' '.join([str(elem) for elem in final_responses_bacnet][1:])
                data = final_responses_bacnet
                for item in data:
                    if item['Label'] == 'No Label':
                        item['Label'] = 'Normal'
                #print(data)
                #data.append(js)
                file.seek(0)
                json.dump(data, file)
                            
        else:
            if len(final_responses_bacnet) == 0:
                pass
            else:
                with open('labelled_bacnet.json', "w") as file:
                    json.dump(final_responses_bacnet, file)

        if os.path.exists('labelled_ntp.json'):
            with open('labelled_ntp.json', "r+") as file:
                data = json.load(file)
                #print(type(data))
                data = data+final_responses_ntp
                for item in data:
                    if item['Label'] == 'No Label':
                        item['Label'] = 'Normal'
                print(data)
                
                #data.append(js)
                file.seek(0)
                json.dump(data, file)
                            
        else:
            if len(final_responses_ntp) == 0:
                pass
            else:
                with open('labelled_ntp.json', "w") as file:
                    json.dump(final_responses_ntp, file)

        if os.path.exists('labelled_radius.json'):
            with open('labelled_radius.json', "r+") as file:
                data = json.load(file)
                #print(type(data))
                data = data+final_responses_radius
                for item in data:
                    if item['Label'] == 'No Label':
                        item['Label'] = 'Normal'
                #print(data)
                
                #data.append(js)
                file.seek(0)
                json.dump(data, file)
                            
        else:
            if len(final_responses_radius) == 0:
                pass
            else:
                with open('labelled_radius.json', "w") as file:
                    json.dump(final_responses_radius, file)
    def  estimate_distance(self, old_data, new_data, protocol):          
        df = pd.concat([old_data, new_data]).reset_index(drop=True)
        #attack_dict = list(list(db['{}_attacks'.format(protocol.lower())].find()[0].items())[1][1].values())[-1]+1
        #attackLabelsToInt = {"Label":attack_dict}
        #df.replace(attackLabelsToInt, inplace = True)
        #df_index = df.index[df['timestamp']==timestmp][0] 
        #if df.iloc[df_index +1:].empty:
            #return -1      
        df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
            'Dst Port'])
            
        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as file:
                timestmp = file.read()
                print(timestmp)
        timestmp = np.int64(int(timestmp))
                
        
        df.to_csv('mqtt_new_df.csv')
            
        df = df.iloc[:,1:]
        df_cols = df.columns
        #print('this is df')
        #print(df)
        scaler = MinMaxScaler()
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df),columns = df_cols)
        #print(df)
        #print(math.exp(0.2))
                    
        integrated_list = []
        for i in range(0, len(df)):
            median_list = []
            sum_elements = 0
                        
                        
            for j in range (0, len(df)):
                            
                #print(format(0.1, '.17f'))
                #print (format((math.exp((-(pow(distance.euclidean
                #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))))
                #print (((-(pow(distance.euclidean
                #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                    
                median_list.append(math.exp((-(pow(distance.euclidean
                (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                sum_elements+=math.exp((-(pow(distance.euclidean
                (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                #print(median_list)
                        
                            
            median_list = [item / sum_elements for item in median_list]
            integrated_list.append(median_list)
            #print(median_list)
            #print(integrated_list)
            #print(len(integrated_list))
        old_list =  [0] * int((len(integrated_list)/2))
        new_list =  [0] * int((len(integrated_list)/2))
        for i in range(0, int((len(integrated_list)/2)-1)):
                        
                #print(integrated_list[i])
                old_list[i]= sum(integrated_list[i])
        for k in range(int(len(integrated_list)/2), len(integrated_list)):
                        
                #print(integrated_list[k])
                new_list[i] = sum(integrated_list[k])
        #print(old_list)
        #print(new_list) 
        old_list = [item/(len(integrated_list)/2) for item in old_list]
        new_list = [item/(len(integrated_list)/2) for item in new_list]
        #print(old_list)
        #print(new_list) 
        difference = []

                    

        zip_object = zip(old_list, new_list)

        for old_list_i, new_list_i in zip_object:

            difference.append(abs(old_list_i-new_list_i))

        #print(difference)

        TVD = sum(difference)*0.5
        print(TVD)
        return TVD
    def estimate_permutated_distance(self, old_data, new_data):
            
            
            
            df = pd.concat([old_data, new_data]).reset_index(drop=True)
            
            df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
            'Dst Port'])
            
            
            df = df.iloc[:,1:]
            df_cols = df.columns
            #print('this is df')
            #print(df)
        
            scaler = MinMaxScaler()
            scaler.fit(df)
            df = pd.DataFrame(scaler.transform(df),columns = df_cols)
            #print(df)
            df1 = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
            #print(df1)
            df = df1.copy()
            
            integrated_list = []
            for i in range(0, len(df)):
                median_list = []
                sum_elements = 0
                
                for j in range (0, len(df)):
                    
                    
                    median_list.append(math.exp((-(pow(distance.euclidean
                    (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                    sum_elements+=math.exp((-(pow(distance.euclidean
                    (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                    
                median_list = [item / sum_elements for item in median_list]
                integrated_list.append(median_list)
            
            old_list =  [0] * int((len(integrated_list)/2))
            new_list =  [0] * int((len(integrated_list)/2))
            for i in range(0, int((len(integrated_list)/2)-1)):
                
                old_list[i]= sum(integrated_list[i])
            for k in range(int(len(integrated_list)/2), len(integrated_list)):
                
                new_list[i] = sum(integrated_list[k])
            
            old_list = [item/(len(integrated_list)/2) for item in old_list]
            new_list = [item/(len(integrated_list)/2) for item in new_list]
            
            difference = []

            

            zip_object = zip(old_list, new_list)

            for old_list_i, new_list_i in zip_object:

                difference.append(abs(old_list_i-new_list_i))

            

            TVD = sum(difference)*0.5
            print(TVD)
            return TVD
    def form_data(self, data_limit):
        #myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        #print('============================================')
        protocols = ['MQTT', 'MODBUS', 'BACNET', 'NTP', 'RADIUS']
        #protocols = ['MQTT']
        self.data_processing('response.json')
        for protocol in protocols:
            if  os.path.exists('labelled_{}.json'.format(protocol.lower())) or os.path.exists('history_{}.json'.format(protocol.lower())):

                result = 'Concept drift detection'

                while result == 'Concept drift detection' or result == 'No concept drift detection':
                    print(result)
                
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        print('exists')
                        with open ('labelled_{}.json'.format(protocol.lower())) as file:
                                    a_json = json. load(file)
                        #print(a_json)
                        dataframe = pd.DataFrame(a_json)
                        
                        df = dataframe.iloc[:,1:]
                        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                        df = df.reset_index(drop = True)
                        
                        print('this is DF from labelled netflows')
                        print(df)
                        
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as file:
                            timestmp = file.read()
                            print('last checkpoint of timestamp')
                            print(timestmp)
                        timestmp = np.int64(int(timestmp))
                        csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                        dataframe = df[df.index[df['timestamp']==timestmp][0]+1:df.index[df['timestamp']==timestmp][0]+data_limit]
                        #csv = df[:df.index[df['timestamp']==timestmp][0]]
                        #dataframe = df[df.index[df['timestamp']==timestmp][0]:]
                        
                        print('XXXXXXXXXXXXXXXXXXXXXXXX')
                        print('before checking')
                        print(csv)
                        print('after checking')
                        print(dataframe)
                        if dataframe.empty:
                            print('empry to dataframe')
                            dataframe = df.iloc[-1].to_frame().transpose()
                            #pass
                        else:
                            pass
                        print(dataframe)
                        if len(dataframe)<= len(csv):
                                    
                            csv = csv.iloc[-len(dataframe):].reset_index(drop=True)
                        else:
                            dataframe = dataframe.iloc[:len(csv)].reset_index(drop=True)
                        
                        
                        timestamp = dataframe.iloc[-1]['timestamp']
                        #csv = csv.iloc[:3]
                        print('**********ITEMS******************')
                        print(type(timestamp))
                        print(timestamp)
                        print(dataframe)
                        print(csv)
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'w') as file:
                            file.write(str(timestamp))
                            print(timestamp)
                        df_index = df.index[df['timestamp']==timestamp][0]####EDW THA MPEI -1
                        #distance = 0
                        
                        if df.iloc[df_index +1:].empty:
                            print('empty')
                            distance = -1
                        else:
                            distance = 0
                            print('not empty')
                        print(df_index)
                        print(distance)
                        print(df.iloc[df_index +1:])
                        #csv = dataframe.iloc[:df_index].reset_index(drop = True)
                        #print(csv)
                        #dataframe = dataframe.iloc[df_index:]#.reset_index(drop = True)
                        print('again dataframe')
                        print(dataframe)
                        #print(dataframe.iloc[df_index])
                        
                        #timestmp = np.fromstring(timestmp, dtype=float)
                        #print(timestmp)
                        #print(type(timestmp))
                        #print(dataframe)
                        #print(csv)
                        columns = []
                        with open ('columns_{}.txt'.format(protocol.lower()), 'r') as cols:
                            colum = cols.read()
                        columns = colum.split(',')[:-1]
                        print(columns)
                            #print(colum)
                        #for item in colum:
                            #columns.append(item)
                        #print(columns)
                        dataframe = dataframe[columns]
                        csv = csv[columns]
                        print('NEW CSV')
                        print(csv)
                        #print(dataframe)
                        with open('{}_attacks.json'.format(protocol.lower()),'r') as file:
                            attackLabelsToInt = json.load(file)
                        dataframe.replace(attackLabelsToInt, inplace = True)
                        csv.replace(attackLabelsToInt, inplace = True)
                        print(dataframe['Label'])
                        
                        #timestamp_dict = {"timestamp":timestamp}
                        #with open ('last_drift_point_{}.txt'.format(protocol.lower()),'w') as file:
                            #file.write(str(timestamp))
                        
                        
                    else:
                        print('not exists')
                        if protocol == "MQTT":
                            csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
                        elif protocol == "BACNET":
                            csv = pd.read_csv("datasets/bacnet_dataset.csv")
                        elif protocol == "MODBUS":
                            csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
                        elif protocol == "NTP":
                            csv = pd.read_csv("datasets/ntp_dataset_1.csv")
                        elif protocol == "RADIUS":
                            csv = pd.read_csv("datasets/radius_dataset.csv")
                        if  os.path.exists('labelled_{}.json'.format(protocol.lower())):
                                with open ('labelled_{}.json'.format(protocol.lower())) as file:
                                    a_json = json. load(file)
                                
                                dataframe = pd.DataFrame(a_json)
                                df = dataframe.iloc[:,1:]
                                print(df)
                                #timestamp = df.iloc[-1]['timestamp']
                                #print(type(timestamp))
                                #print(timestamp)
                                #timestamp_dict = {"timestamp":timestamp}
                                #with open ('last_drift_point_{}.txt'.format(protocol.lower()),'w') as file:
                                        #file.write(str(timestamp))
                                #df_index = dataframe.index[dataframe['timestamp']==timestamp][0]#Edw tha mpei -1
                                print('==============================')
                                
                                csv = csv.iloc[:3]
                                print(csv)
                                if len(dataframe)<= len(csv):
                                    
                                    csv = csv.iloc[-len(dataframe):].reset_index(drop=True)
                                else:
                                    dataframe = dataframe.iloc[:len(csv)].reset_index(drop=True)
                                print('This is dataframe')
                                print(dataframe)
                                print(csv)
                                timestamp = dataframe.iloc[-1]['timestamp']
                                print(type(timestamp))
                                print(timestamp)
                                df_index = dataframe.index[dataframe['timestamp']==timestamp][0]#Edw tha mpei -1
                                #timestamp = dataframe.iloc[-1]['timestamp']
                                print(type(timestamp))
                                print(timestamp)
                                timestamp_dict = {"timestamp":timestamp}
                                with open ('last_check_point_{}.txt'.format(protocol.lower()),'w') as file:
                                        file.write(str(timestamp))
                                print(dataframe)
                                columns = list(set(dataframe.columns).intersection(csv.columns))
                            
                                with open ('columns_{}.txt'.format(protocol.lower()), 'w') as cols:
                                    for col in columns:
                                        cols.write(col+ ",")
                                
                                #if df.iloc[df_index +1:].empty:
                                        #return -1
                                dataframe = dataframe[columns]
                                if dataframe.iloc[df_index +1:].empty:
                                    print('empty')
                                    distance = 0
                                else:
                                    distance = 0
                                    print('not empty')
                                print(df_index)
                                #print(dataframe)
                                if  os.path.exists('{}_attacks.json'.format(protocol.lower())):
                                    with open('{}_attacks.json'.format(protocol.lower()),'r') as file:
                                        attackLabelsToInt = json.load(file)
                                else:
                                    attackLabelsToInt = {
                                    "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    with open('{}_attacks.json'.format(protocol.lower()),'w') as file:
                                                    json.dump(attackLabelsToInt, file)
                                csv.replace(attackLabelsToInt, inplace = True)
                                dataframe.replace(attackLabelsToInt, inplace = True)
                                
                    print(dataframe['Label'])
                    print(csv['Label'])
                    print('00000000000000000000000')
                    if distance != -1:
                        distance = self.estimate_distance(csv, dataframe, protocol = protocol)
                        print('!!!!!!!!!!!!!!!!!!!!!!!!')
                        print (distance)
                    
                        permutations = 100
                        perm_list = []
                        for i in range(0, permutations):
                            perm_list.append(self.estimate_permutated_distance(csv, dataframe))
                        print(perm_list)
                        critical = (np.percentile(perm_list, 95))
                        print(distance)
                        print(critical)
                        result = ''
                        if distance > critical:
                            #print("Concept drift detection") 
                            result = 'Concept drift detection'
                            with open ('last_drift_point_{}.txt'.format(protocol.lower()),'w') as file:
                                file.write(str(timestamp))
                        else:
                            #print("No concept drift detection")
                            result = 'No concept drift detection'
                            self.get_user_data(protocol = protocol)
                            #form_data(data_limit = 5)
                        print('RESULT IS ____________')
                        print(result)
                    else:
                        #print('There are not new data')
                        result = 'There are not new data'
                    print('RESULT IS ____________')
                    print(result)
            else:
                pass  
            break

    #form_data(data_limit = 5)
    def get_user_data(self,protocol):
        
        #request_body = request.get_json()
        #with open('request_body.json') as file:
            #request_body = json.load(file)
        
        #data_processing('response.json')
        protocols = []
        netflow_ids = []
        labels = []
        #for item in request_body:
            #netflow_ids.append(item['value'])
            #labels.append(item["Label"])
        
        user_list = list(zip(netflow_ids, labels))
        
        all_protocols = ['MQTT','MODBUS','BACNET','NTP',"RADIUS"]
        mqtt = []
        modbus = []
        bacnet = []
        ntp = []
        radius = []
        x = pd.DataFrame()
        y = pd.DataFrame()
        z = pd.DataFrame()
        a = pd.DataFrame()
        b = pd.DataFrame()
        x1 = pd.DataFrame()
        y1 = pd.DataFrame()
        z1 = pd.DataFrame()
        a1 = pd.DataFrame()
        b1 = pd.DataFrame()
        
        
        #for protocol in all_protocols:
        print('INSIDE GET USER DATA')
        print(protocol)
        if protocol == 'MQTT':
            if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                with open ('{}_attacks.json'.format(protocol.lower())) as file:
                                    
                    attackLabelsToInt = json.load(file)
                    print('*********************')
                    print('attacks is')
                    print(attackLabelsToInt)
                    print(type(attackLabelsToInt))
                    targets = list(list(attackLabelsToInt.items())[0][1].keys())
                                    
                    num_classes = len(list(list(attackLabelsToInt.items())[0][1].keys()))
            else:
                    attackLabelsToInt = {
                    "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                        
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    targets = len(list(list(attackLabelsToInt.values())[0].values()))
            if os.path.exists('history_{}.json'.format(protocol.lower())):
                    
                with open('history_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                        
                    mqtt_user = json.load(file)#[1:-1].split('},')
                    mqtt_user =json.loads(mqtt_user)
                    #print(mqtt_user)
                    x = pd.DataFrame.from_records(mqtt_user)
                    
            if os.path.exists('labelled_{}.json'.format(protocol.lower())):
                    
                with open('labelled_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                    #if protocol == 'MQTT':
                    mqtt_label = json.load(file)
                    #print(type(mqtt))
                    x1 = pd.DataFrame(mqtt_label)
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as driftfile:
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
            if x.empty and x1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
                mqtt_final = pd.concat([x, x1])
                print(x)
                print(x1)
                print(mqtt_final)
                columns  = list(set(mqtt_final.columns).intersection(csv.columns))
                mqtt_final = pd.concat([csv,mqtt_final[columns]]).reset_index(drop=True)
                mqtt_final.replace(attackLabelsToInt, inplace = True)
                print('train mqtt model')
                print(mqtt_final)
                self.TrainModels(protocol = 'MQTT', attackLabelsToInt = attackLabelsToInt, input_data=mqtt_final)
                        
                    
                    
        #attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}            
        elif protocol == 'BACNET':
            if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                with open ('{}_attacks.json'.format(protocol.lower())) as file:
                                    
                    attackLabelsToInt = json.load(file)
                    print('*********************')
                    print('attacks is')
                    print(attackLabelsToInt)
                    print(type(attackLabelsToInt))
                    targets = list(list(attackLabelsToInt.items())[0][1].keys())
                                    
                    num_classes = len(list(list(attackLabelsToInt.items())[0][1].keys()))
            else:
                    attackLabelsToInt = {"Label": {"Normal": 0, "BACnet Fuzzing": 1, "Tampering": 2, "Flooding": 3}}
                        
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    targets = len(list(list(attackLabelsToInt.values())[0].values()))
            if os.path.exists('history_{}.json'.format(protocol.lower())):
                    
                with open('history_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                        
                    bacnet_user = json.load(file)#[1:-1].split('},')
                    bacnet_user =json.loads(bacnet_user)
                    #print(mqtt_user)
                    y = pd.DataFrame.from_records(bacnet_user)
                    
            if os.path.exists('labelled_{}.json'.format(protocol.lower())):
                    
                with open('labelled_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                    #if protocol == 'MQTT':
                    bacnet_label = json.load(file)
                    #print(type(mqtt))
                    y1 = pd.DataFrame(bacnet_label)
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as driftfile:
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            y1 = y1[y1.index[y1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
            if y.empty and y1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/bacnet_dataset.csv")
                bacnet_final = pd.concat([y, y1])
                columns  = list(set(bacnet_final.columns).intersection(csv.columns))
                bacne_final = pd.concat([csv,bacnet_final[columns]]).reset_index(drop=True)
                #mqtt_final.replace(attackLabelsToInt, inplace = True)
                bacnet_final.replace(attackLabelsToInt, inplace = True)
                print('train bacnet model')
                self.TrainModels(protocol = 'BACNET', attackLabelsToInt = attackLabelsToInt, input_data=bacnet_final)
        elif protocol == 'MODBUS':
            if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                with open ('{}_attacks.json'.format(protocol.lower())) as file:
                                            
                    attackLabelsToInt = json.load(file)
                    print('*********************')
                    print('attacks is')
                    print(attackLabelsToInt)
                    print(type(attackLabelsToInt))
                    targets = list(list(attackLabelsToInt.items())[0][1].keys())
                                            
                    num_classes = len(list(list(attackLabelsToInt.items())[0][1].keys()))
            else:
                    attackLabelsToInt = {
                                        "Label": {'Normal':0, 'UID brute force':1, 'Enumeration Function':2, 'Fuzzing Read Holding Registers':3}}                
                                
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    targets = len(list(list(attackLabelsToInt.values())[0].values()))
            if os.path.exists('history_{}.json'.format(protocol.lower())):
                    
                with open('history_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                        
                    modbus_user = json.load(file)#[1:-1].split('},')
                    modbus_user =json.loads(modbus_user)
                    #print(mqtt_user)
                    z = pd.DataFrame.from_records(modbus_user)
                    
            if os.path.exists('labelled_{}.json'.format(protocol.lower())):
                    
                with open('labelled_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                    #if protocol == 'MQTT':
                    modbus_label = json.load(file)
                    #print(type(mqtt))
                    z1 = pd.DataFrame(modbus_label)
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as driftfile:
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            z1 = z1[z1.index[z1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
            if z.empty and z1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
                modbus_final = pd.concat([z, z1])
                columns  = list(set(modbus_final.columns).intersection(csv.columns))
                modbus_final = pd.concat([csv,modbus_final[columns]]).reset_index(drop=True)
                modbus_final.replace(attackLabelsToInt, inplace = True)
                print('train modbus model')
                self.TrainModels(protocol = 'MODBUS', attackLabelsToInt = attackLabelsToInt, input_data=modbus_final)          
                        
                        
        #attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}                
        elif protocol == 'NTP':
            if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                with open ('{}_attacks.json'.format(protocol.lower())) as file:
                                            
                    attackLabelsToInt = json.load(file)
                    print('*********************')
                    print('attacks is')
                    print(attackLabelsToInt)
                    print(type(attackLabelsToInt))
                    targets = list(list(attackLabelsToInt.items())[0][1].keys())
                                            
                    num_classes = len(list(list(attackLabelsToInt.items())[0][1].keys()))
            else:
                    attackLabelsToInt = {"Label": {"Normal": 0, "KissOfDeath": 1, "TimeSkimming": 2}}                
                                
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    targets = len(list(list(attackLabelsToInt.values())[0].values()))
            if os.path.exists('history_{}.json'.format(protocol.lower())):
                    
                with open('history_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                        
                    ntp_user = json.load(file)#[1:-1].split('},')
                    ntp_user =json.loads(ntp_user)
                    #print(mqtt_user)
                    a = pd.DataFrame.from_records(ntp_user)
                    
            if os.path.exists('labelled_{}.json'.format(protocol.lower())):
                    
                with open('labelled_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                    #if protocol == 'MQTT':
                    ntp_label = json.load(file)
                    #print(type(mqtt))
                    a1 = pd.DataFrame(ntp_label)
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as driftfile:
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            a1 = a1[a1.index[a1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
        

            if a.empty and a1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/ntp_dataset_1.csv")
                ntp_final = pd.concat([a, a1])
                columns  = list(set(ntp_final.columns).intersection(ntp.columns))
                ntp_final = pd.concat([ntp,ntp_final[columns]]).reset_index(drop=True)
                print('train ntp model')
                    
                ntp_final.replace(attackLabelsToInt, inplace = True)
                self.TrainModels(protocol = 'NTP', attackLabelsToInt = attackLabelsToInt, input_data=ntp_final)
        
        elif protocol == 'RADIUS':
            if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                with open ('{}_attacks.json'.format(protocol.lower())) as file:
                                            
                    attackLabelsToInt = json.load(file)
                    print('*********************')
                    print('attacks is')
                    print(attackLabelsToInt)
                    print(type(attackLabelsToInt))
                    targets = list(list(attackLabelsToInt.items())[0][1].keys())
                                            
                    num_classes = len(list(list(attackLabelsToInt.items())[0][1].keys()))
            else:
                    attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}                
                                
                    num_classes = len(list(list(attackLabelsToInt.values())[0].values()))
                    targets = len(list(list(attackLabelsToInt.values())[0].values()))
            if os.path.exists('history_{}.json'.format(protocol.lower())):
                    
                with open('history_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                        
                    radius_user = json.load(file)#[1:-1].split('},')
                    radius_user =json.loads(radius_user)
                    #print(mqtt_user)
                    b = pd.DataFrame.from_records(radius_user)
                    
            if os.path.exists('labelled_{}.json'.format(protocol.lower())):
                    
                with open('labelled_{}.json'.format(protocol.lower())) as file :
                    #print('netflows_{}.json'.format(protocol))
                    #if protocol == 'MQTT':
                    radius_label = json.load(file)
                    #print(type(mqtt))
                    b1 = pd.DataFrame(radius_label)
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.txt'.format(protocol.lower())):
                        with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as driftfile:
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            b1 = b1[b1.index[b1['timestamp']==timestmp][0]:].reset_index(drop = True)
            if b.empty and b1.empty:
            
                pass
            else:
                csv = pd.read_csv("datasets/radius_dataset.csv")
                radius_final = pd.concat([b, b1])
                columns  = list(set(radius_final.columns).intersection(csv.columns))
                radius_final = pd.concat([csv,radius_final[columns]]).reset_index(drop=True)
                if os.path.exists('{}_attacks.json'.format(protocol.lower())):
                    with open ('{}_attacks.json'.format(protocol.lower())) as file:
                        attackLabelsToInt = json.load(file)
                else:
                    attackLabelsToInt = {"Label": {"Normal": 0, "Brute Force": 1}}
                
                radius_final.replace(attackLabelsToInt, inplace = True)
                print('train radius model')
                self.TrainModels(protocol = 'RADIUS', attackLabelsToInt = attackLabelsToInt, input_data=radius_final)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                #print('There are not Data')
        #columns  = list(set(dataframe.columns).intersection(csv.columns))
        
        
        
        
        


    def TrainModels(self,protocol, attackLabelsToInt, input_data):#,  attack
        #As we make the same training process we hard-coded the request
        #for training in parallel process
        print('INSIDE TRAIN MODESL')
        protocol = protocol
        
        modelTypes= ["SDAE", "KNN","LogReg", "Random-Forest","SVC"]
        #modelTypes= ["SDAE","KNN", "SVC"]
        
        crossVal = 2
        
        
        #request_body = request.get_json()
        
        #with open('request_body.json') as file:
            #request_body = json.load(file) 
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
        
        
        
        self.train(input_data, protocol, modelTypes, crossVal)
        #train_task = Process(target=train, 
        #args=(input_data, protocol, modelTypes, crossVal))#,  attack
        
        #train_task.start()
        #return 'Model re-training  started'    
    def train(self, data, protocol, models, cv):#,  attack
        # register SPARK backend
        register_spark()

        start_time = time.time()
        
        results = Parallel(backend='spark', n_jobs=len(models))(
            delayed(self.parallelTrain)(data, protocol, model, cv) for model in models)#, attack

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


    def parallelTrain(self,csv, protocol, model_type, cv):#,  attack
        
        data = Data(csv, protocol, model_type)
        #It is called the training module
        x_train, x_test, y_train, y_test = data.train_test_data()
        #if model_type != 'SDAE':
            #pass
        #else:
        train = Training(x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), data, protocol, model_type, cv)#, attack
        #(self, x_train, y_train, x_test, y_test, data, protocol, model_type, cv)       
        
        report = train.classify()
        print('F1 score for ', model_type , ': ' ,  report.get('weighted avg').get('f1-score'))
        f1 = report.get('macro avg').get('f1-score')
        return model_type, f1


if __name__ == "__main__":
    concept_drift_odj = concept_drift()
    concept_drift_odj.form_data(data_limit = 5)


#get_user_data()
#data_processing('response.json')









