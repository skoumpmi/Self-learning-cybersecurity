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
from retraining import Training_ML, Training_DL
from training import Training
#from retraining_exp import Training_ML, Training_DL
#from training_exp import Training
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


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
#myclient = pymongo.MongoClient("mongodb://localhost:5001/")
db = myclient["labelled_netflows_db"]

def form_data(data_limit):
        #myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        #print('============================================')
        protocols = ['MQTT', 'MODBUS']
        #protocols = ['MQTT']
        ####self.data_processing('response.json')
        for protocol in protocols:
            print('==================================')
            print('==================================')
            print(protocol)
            print('==================================')
            print('==================================')
            if 'labelled_{}'.format(protocol.lower()) in db.list_collection_names():
            #if  os.path.exists('labelled_{}.json'.format(protocol.lower())) or os.path.exists('history_{}.json'.format(protocol.lower())):

                result = 'Concept drift detection'

                while result == 'Concept drift detection' or result == 'No concept drift detection':
                    print(result)
                    if 'last_check_point_{}'.format(protocol.lower()) in db.list_collection_names():
                    #if  os.path.exists('last_check_point_{}.json'.format(protocol.lower())):
                        print('exists')
                        #with open ('labelled_{}.json'.format(protocol.lower())) as file:
                                    #a_json = json. load(file)
                        #print(a_json)
                        #dataframe = pd.DataFrame(a_json)
                        dataframe = pd.DataFrame(list(db["labelled_{}".format(protocol.lower())].find()))
                        df = dataframe.iloc[:,1:]
                        #df.drop_duplicates(subset=subset=['A', 'B'], keep='first', inplace=True)
                        #df = df.reset_index(drop = True)
                        ###df.drop_duplicates(subset=['Flow ID', 'timestamp'], keep='first', inplace=True)
                        print('this is DF from labelled netflows')
                        print(df)
                        r'''
                        with open ('last_check_point_{}.json'.format(protocol.lower()),'r') as file:
                            #timestmp = file.read()
                            timestmpp = json. load(file)
                            print('last checkpoint of timestamp')
                            print(timestmpp)
                        timestmp = np.int64(int(timestmpp.get('timestamp')))
                        flowid = timestmpp.get('Flow ID')
                        '''

                        print('================================================================')
                        last_check_var = (list(list(db['last_check_point_{}'.format(protocol.lower())].find())[0].items()))[1:]
                        print(last_check_var)
                        timestmp = last_check_var[1][1]
                        print(type(timestmp))
                        timestmp = np.int64((timestmp))
                        print(type(timestmp))
                        print(timestmp)
                        flowid = last_check_var[0][1]
                        
                        print(flowid)
                        print(type(flowid))
                        #dict.get('Age')
                        #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                        #dataframe = df[df.index[df['timestamp']==timestmp][0]+1:df.index[df['timestamp']==timestmp][0]+data_limit]
                        csv = df[:df.index[(df['timestamp']==timestmp)&(df['Flow ID']==flowid)][0]+1]
                        dataframe = df[df.index[(df['timestamp']==timestmp)&(df['Flow ID']==flowid)][0]+1:df.index[(df['timestamp']==timestmp)&(df['Flow ID']==flowid)][-1]+data_limit]
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
                        flowid = dataframe.iloc[-1]['Flow ID']
                        #csv = csv.iloc[:3]
                        print('**********ITEMS******************')
                        print(type(timestamp))
                        print(timestamp)
                        print(dataframe)
                        print(csv)
                        timestamp_dict = {"flowid":flowid,"timestamp":timestamp}
                        #with open ('{}_attacks.json'.format(unique_attacks[i][0].lower()),'w') as outfile:
                        if 'last_check_point_{}'.format(protocol.lower()) in db.list_collection_names():
                            db['last_check_point_{}'.format(protocol.lower())].drop()
                                
                        try:
                            db['last_check_point_{}'.format(protocol.lower())].insert_one(timestamp_dict)
                        except pymongo.errors.InvalidDocument:
                                # Python 2.7.10 on Windows and Pymongo are not forgiving
                                # If you have foreign data types you have to convert them
                                    n = {}
                                    for k, v in timestamp_dict.items():
                                
                                        if isinstance(v, np.int64):
                                            v = int(v)
                                        if isinstance(v, np.float64):
                                            v = float(v)

                                        n[k] = v
                        
                                    db['last_check_point_{}'.format(protocol.lower())].insert_one(n)
                        #{"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}
                        r'''
                        with open ('last_check_point_{}.json'.format(protocol.lower()),'w') as file:
                                #file.write(str(timestamp))
                                json.dump({'Flow ID':'{}'.format(flowid),'timestamp':'{}'.format(str(timestamp))},file)
                                #file.write({'Flow ID':'{}'.format(flowid),'timestamp':'{}'.format(str(timestamp))})
                        '''
                        
                        
                        
                        #with open ('last_check_point_{}.txt'.format(protocol.lower()),'w') as file:
                            #file.write({'Flow ID':'{}'.format(flowid),'timestamp':'{}'.format(str(timestamp))})
                            #print(timestamp)
                        ##df_index = df.index[(df['timestamp']==timestamp &df['Flow ID']==flowid)]#[0]####EDW THA MPEI -1
                        df_index = df.index[(df['timestamp']==timestamp) & (df['Flow ID']==flowid)][0]
                        print(df_index)
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
                        r'''
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
                        '''
                        print('NEW CSV')
                        print(csv)
                        #print(dataframe)
                        if '{}_attacks'.format(protocol.lower()) in db.list_collection_names():
                                    
                                #if  os.path.exists('{}_attacks.json'.format(protocol.lower())):
                                #with open('{}_attacks.json'.format(protocol.lower()),'r') as file:
                                    #attackLabelsToInt = json.load(file)
                            attackLabelsToInt = {"Label": list(list(db['{}_attacks'.format(protocol.lower())].find())[0].items())[1][1]}
                        else:
                                    if protocol == 'MQTT':
                                             attackLabelsToInt = {
                                            "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    if protocol == 'MODBUS':
                                        attackLabelsToInt = {
                                             "Label": {'Normal': 0, 'UID brute force': 1, 'Enumeration Function': 2,
                                                'Fuzzing Read Holding Registers': 3}}
                        r'''
                        with open('{}_attacks.json'.format(protocol.lower()),'r') as file:
                            attackLabelsToInt = json.load(file)
                        '''
                        dataframe.replace(attackLabelsToInt, inplace = True)
                        csv.replace(attackLabelsToInt, inplace = True)
                        
                        print(dataframe['Label'])
                        
                        #timestamp_dict = {"timestamp":timestamp}
                        #with open ('last_drift_point_{}.txt'.format(protocol.lower()),'w') as file:
                            #file.write(str(timestamp))
                        
                        
                    else:
                        print('not exists')
                        csv = pd.read_csv("datasets/{}_dataset.csv".format(protocol.lower()))
                        r'''
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
                        '''
                        if 'labelled_{}'.format(protocol.lower()) in db.list_collection_names():
                        #if  os.path.exists('labelled_{}.json'.format(protocol.lower())):
                                #with open ('labelled_{}.json'.format(protocol.lower())) as file:
                                    #a_json = json. load(file)
                                
                                #dataframe = pd.DataFrame(a_json)
                                dataframe = pd.DataFrame(list(db["labelled_{}".format(protocol.lower())].find()))
                                #dataframe = pd.DataFrame(a_json)
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
                                flowid = dataframe.iloc[-1]['Flow ID']
                                print(type(timestamp))
                                print(timestamp)
                                df_index = dataframe.index[dataframe['timestamp']==timestamp][0]#Edw tha mpei -1
                                #timestamp = dataframe.iloc[-1]['timestamp']
                                print(type(timestamp))
                                print(timestamp)
                                timestamp_dict = {"flowid":flowid,"timestamp":timestamp}
                                #with open ('{}_attacks.json'.format(unique_attacks[i][0].lower()),'w') as outfile:
                                if 'last_check_point_{}'.format(protocol.lower()) in db.list_collection_names():
                                    db['last_check_point_{}'.format(protocol.lower())].drop()
                                
                                try:
                                    db['last_check_point_{}'.format(protocol.lower())].insert_one(timestamp_dict)
                                except pymongo.errors.InvalidDocument:
                                # Python 2.7.10 on Windows and Pymongo are not forgiving
                                # If you have foreign data types you have to convert them
                                    n = {}
                                    for k, v in timestamp_dict.items():
                                
                                        if isinstance(v, np.int64):
                                            v = int(v)
                                        if isinstance(v, np.float64):
                                            v = float(v)

                                        n[k] = v
                        
                                    db['last_check_point_{}'.format(protocol.lower())].insert_one(n)
                                ##with open ('last_check_point_{}.json'.format(protocol.lower()),'w') as file:
                                        #file.write(str(timestamp))
                                        ##json.dump({'Flow ID':'{}'.format(flowid),'timestamp':'{}'.format(str(timestamp))},file)
                                        #file.write({'Flow ID':'{}'.format(flowid),'timestamp':'{}'.format(str(timestamp))})

                                print(dataframe)
                                columns = list(set(dataframe.columns).intersection(csv.columns))
                            
                                #with open ('columns_{}.txt'.format(protocol.lower()), 'w') as cols:
                                    #for col in columns:
                                        #cols.write(col+ ",")
                                
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
                                if '{}_attacks'.format(protocol.lower()) in db.list_collection_names():
                                    
                                #if  os.path.exists('{}_attacks.json'.format(protocol.lower())):
                                    #with open('{}_attacks.json'.format(protocol.lower()),'r') as file:
                                        #attackLabelsToInt = json.load(file)
                                    attackLabelsToInt = {"Label": list(list(db['{}_attacks'.format(protocol.lower())].find())[0].items())[1][1]}
                                else:
                                    if protocol == 'MQTT':
                                             attackLabelsToInt = {
                                            "Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
                                    if protocol == 'MODBUS':
                                        attackLabelsToInt = {
                                             "Label": {'Normal': 0, 'UID brute force': 1, 'Enumeration Function': 2,
                                                'Fuzzing Read Holding Registers': 3}}
                                    #with open('{}_attacks.json'.format(protocol.lower()),'w') as file:
                                                    #json.dump(attackLabelsToInt, file)
                                    mycol = db["{}_attacks".format(protocol.lower())].insert_one(attackLabelsToInt)
                                csv.replace(attackLabelsToInt, inplace = True)
                                dataframe.replace(attackLabelsToInt, inplace = True)
                                
                    print(dataframe['Label'])
                    print(csv['Label'])
                    print('00000000000000000000000')
                    if distance != -1:
                        distance = estimate_distance(csv, dataframe, protocol = protocol)
                        print('!!!!!!!!!!!!!!!!!!!!!!!!')
                        print (distance)
                    
                        permutations = 100
                        perm_list = []
                        for i in range(0, permutations):
                            perm_list.append(estimate_permutated_distance(csv, dataframe))
                        print(perm_list)
                        critical = (np.percentile(perm_list, 95))
                        print(distance)
                        print(critical)
                        result = ''
                        if distance > critical:
                            #print("Concept drift detection") 
                            result = 'Concept drift detection'
                            with open ('last_drift_point_{}.txt'.format(protocol.lower()),'a') as file:
                                file.write(str(timestamp))
                        else:
                            #print("No concept drift detection")
                            result = 'No concept drift detection'
                            ###############################get_user_data(protocol = protocol)
                            ###self.get_user_data(protocol = protocol)
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
            #break
def  estimate_distance( old_data, new_data, protocol):          
        df = pd.concat([old_data, new_data]).reset_index(drop=True)
        #attack_dict = list(list(db['{}_attacks'.format(protocol.lower())].find()[0].items())[1][1].values())[-1]+1
        #attackLabelsToInt = {"Label":attack_dict}
        #df.replace(attackLabelsToInt, inplace = True)
        #df_index = df.index[df['timestamp']==timestmp][0] 
        #if df.iloc[df_index +1:].empty:
            #return -1      
        df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
            'Dst Port'])
            
        #with open ('last_check_point_{}.txt'.format(protocol.lower()),'r') as file:
        print('================================================================')
        last_check_var = (list(list(db['last_check_point_{}'.format(protocol.lower())].find())[0].items()))[1:]
        print(last_check_var)
        timestmp = last_check_var[0][1]
        print(type(timestmp))
        print(timestmp)
        flowid = last_check_var[1][1]
        
        print(flowid)
        print(type(flowid))

        r'''
        with open ('last_check_point_{}.json'.format(protocol.lower()),'r') as file:
                timestmppp = json.load(file)
                #timestmp = file.read()
                print(timestmppp)
                print(type(timestmppp))
        #timestmp = np.int64(int(timestmp))
        print('***********TIMESTAMP**************************8')
        timestmp = timestmppp.get('timestamp')
        print(type(timestmp))  
        '''       
        
        df.to_csv('mqtt_new_df.csv')
            
        df = df.iloc[:,1:].reset_index(drop = True).fillna(0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        #df = df.drop(df.columns[np.isnan(df).any()], axis=1).reset_index()
        df_cols = list(df.columns)#.remove('str_protocol').remove('machine')#.remove('str_protocol').remove('machine')
        if 'str_protocol' in df_cols:
            df_cols.remove('str_protocol')
        if 'machine' in df_cols:
            df_cols.remove('machine')
        print(df_cols)
        print(type(df_cols))
        for col in df_cols:
            print(col)
            print(type(df.iloc[0][col]))
        print('this is df')
        print(df)
        df = df[df_cols]
        scaler = MinMaxScaler()
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df),columns = df_cols)
        
        #print(df)
        #print(math.exp(0.2))
                    
        integrated_list = []
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(0)
        
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
        
        r'''
        for i in range(0, len(df)):
                median_list = []
                sum_elements = 0
                
                #for j in range (0, len(df)):
                median_list = [(math.exp((-(pow(distance.euclidean
                (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))for
                j in range (0, len(df))] 
                sum_elements = sum(median_list)  
                    
                    #median_list.append(math.exp((-(pow(distance.euclidean
                    #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                    #sum_elements+=math.exp((-(pow(distance.euclidean
                    #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                    
                median_list = [item / sum_elements for item in median_list]
                integrated_list.append(median_list)
                
                #print(median_list)
                #print(integrated_list)
                #print(sum_elements)
        
        with open ('results_old.txt', 'a') as file:
                file.write('====================================='+'\n') 
                file.write(str(median_list)+'\n')
                file.write(str(integrated_list)+'\n') 
                file.write(str(sum_elements)+'\n') 
                file.write('====================================='+'\n')  
        '''
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
def estimate_permutated_distance( old_data, new_data):
            
            
            
            df = pd.concat([old_data, new_data]).reset_index(drop=True)
            
            df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
            'Dst Port'])
            
            
            df = df.iloc[:,1:]
            #df_cols = df.columns
            #print('this is df')
            #print(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_cols = list(df.columns)#.remove('str_protocol').remove('machine')#.remove('str_protocol').remove('machine')
            if 'str_protocol' in df_cols:
                df_cols.remove('str_protocol')
            if 'machine' in df_cols:
                df_cols.remove('machine')
            print(df_cols)
            print(type(df_cols))
            for col in df_cols:
                print(col)
                print(type(df.iloc[0][col]))
            print('this is df')
            print(df)
            df = df[df_cols]
            scaler = MinMaxScaler()
            scaler.fit(df)
            df = pd.DataFrame(scaler.transform(df),columns = df_cols)
            #print(df)
            df1 = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
            #print(df1)
            df = df1.copy()
            
            integrated_list = []
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.fillna(0)
            for i in range(0, len(df)):
                median_list = []
                sum_elements = 0
                
                #for j in range (0, len(df)):
                median_list = [(math.exp((-(pow(distance.euclidean
                (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))for
                j in range (0, len(df))] 
                sum_elements = sum(median_list)  
                    
                    #median_list.append(math.exp((-(pow(distance.euclidean
                    #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                    #sum_elements+=math.exp((-(pow(distance.euclidean
                    #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                    
                median_list = [item / sum_elements for item in median_list]
                integrated_list.append(median_list)
                
                #print(median_list)
                #print(integrated_list)
                #print(sum_elements)
            #with open ('results_new.txt', 'a') as file:
                #file.write('====================================='+'\n') 
                #file.write(str(median_list)+'\n')
                #file.write(str(integrated_list)+'\n') 
                #file.write(str(sum_elements)+'\n') 
                #file.write('====================================='+'\n')  
            
            
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
#form_data(data_limit = 5)
def get_user_data(protocol):
        
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
                    print(x1.columns)
                    print(x1['timestamp'])
                    #print(x1)
                    if  os.path.exists('last_check_point_{}.json'.format(protocol.lower())):
                        with open ('last_check_point_{}.json'.format(protocol.lower()),'r') as driftfile:
                            point = json.load(driftfile)
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print(point)
                            print(type(point))
                            #print(timestmp)
                            timestmp = point.get('timestamp')#['timestamp']
                            flow = point.get('Flow ID')
                            print(timestmp)
                            print(flow)
                            timestmp = np.int64(int(timestmp))
                            #timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            #x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = x1[x1.index[x1['timestamp']==timestmp]]#.reset_index(drop = True)
                            #print(x1)
                            #x2 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)]]#.reset_index(drop = True)
                            #surveys_df[(surveys_df.year >= 1980) & (surveys_df.year <= 1985)]

                            x1 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)][0]:].reset_index(drop = True)
                            
                            print(x1)
                            #df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                            #####x1 = x1.drop_duplicates().reset_index(drop = True)
                            print(x1)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:] 
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            r'''
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                            '''
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
                ##self.TrainModels(protocol = 'MQTT', attackLabelsToInt = attackLabelsToInt, input_data=mqtt_final)
                        
                    
                    
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
                            point = json.load(driftfile)
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print(point)
                            print(type(point))
                            #print(timestmp)
                            timestmp = point.get('timestamp')#['timestamp']
                            flow = point.get('Flow ID')
                            print(timestmp)
                            print(flow)
                            timestmp = np.int64(int(timestmp))
                            #timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            #x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = x1[x1.index[x1['timestamp']==timestmp]]#.reset_index(drop = True)
                            #print(x1)
                            #x2 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)]]#.reset_index(drop = True)
                            #surveys_df[(surveys_df.year >= 1980) & (surveys_df.year <= 1985)]

                            y1 = y1[y1.index[(y1['timestamp']==timestmp)&(y1['Flow ID'] == flow)][0]:].reset_index(drop = True)
                            
                            print(y1)
                            #df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                            #####y1 = y1.drop_duplicates().reset_index(drop = True)
                            print(y1)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:] 
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            r'''
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                            '''
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
                ##self.TrainModels(protocol = 'BACNET', attackLabelsToInt = attackLabelsToInt, input_data=bacnet_final)
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
                            
                            point = json.load(driftfile)
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print(point)
                            print(type(point))
                            #print(timestmp)
                            timestmp = point.get('timestamp')#['timestamp']
                            flow = point.get('Flow ID')
                            print(timestmp)
                            print(flow)
                            timestmp = np.int64(int(timestmp))
                            #timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            #x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = x1[x1.index[x1['timestamp']==timestmp]]#.reset_index(drop = True)
                            #print(x1)
                            #x2 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)]]#.reset_index(drop = True)
                            #surveys_df[(surveys_df.year >= 1980) & (surveys_df.year <= 1985)]

                            z1 = z1[z1.index[(z1['timestamp']==timestmp)&(z1['Flow ID'] == flow)][0]:].reset_index(drop = True)
                            
                            print(x1)
                            #df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                            z1 = z1.drop_duplicates().reset_index(drop = True)
                            print(z1)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:] 
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            r'''
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                            '''
            if z.empty and z1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/modbus_dataset_26_30_03_2020.csv")
                modbus_final = pd.concat([z, z1])
                columns  = list(set(modbus_final.columns).intersection(csv.columns))
                modbus_final = pd.concat([csv,modbus_final[columns]]).reset_index(drop=True)
                modbus_final.replace(attackLabelsToInt, inplace = True)
                print('train modbus model')
                ##self.TrainModels(protocol = 'MODBUS', attackLabelsToInt = attackLabelsToInt, input_data=modbus_final)          
                        
                        
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
                            point = json.load(driftfile)
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print(point)
                            print(type(point))
                            #print(timestmp)
                            timestmp = point.get('timestamp')#['timestamp']
                            flow = point.get('Flow ID')
                            print(timestmp)
                            print(flow)
                            timestmp = np.int64(int(timestmp))
                            #timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            #x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = x1[x1.index[x1['timestamp']==timestmp]]#.reset_index(drop = True)
                            #print(x1)
                            #x2 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)]]#.reset_index(drop = True)
                            #surveys_df[(surveys_df.year >= 1980) & (surveys_df.year <= 1985)]

                            a1 = a1[a1.index[(a1['timestamp']==timestmp)&(a1['Flow ID'] == flow)][0]:].reset_index(drop = True)
                            
                            print(x1)
                            #df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                            a1 = a1.drop_duplicates().reset_index(drop = True)
                            print(a1)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:] 
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            r'''
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                            '''
        

            if a.empty and a1.empty:
                pass
            else:
                csv = pd.read_csv("datasets/ntp_dataset_1.csv")
                ntp_final = pd.concat([a, a1])
                columns  = list(set(ntp_final.columns).intersection(ntp.columns))
                ntp_final = pd.concat([ntp,ntp_final[columns]]).reset_index(drop=True)
                print('train ntp model')
                    
                ntp_final.replace(attackLabelsToInt, inplace = True)
                ##self.TrainModels(protocol = 'NTP', attackLabelsToInt = attackLabelsToInt, input_data=ntp_final)
        
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
                            point = json.load(driftfile)
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print(point)
                            print(type(point))
                            #print(timestmp)
                            timestmp = point.get('timestamp')#['timestamp']
                            flow = point.get('Flow ID')
                            print(timestmp)
                            print(flow)
                            timestmp = np.int64(int(timestmp))
                            #timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            #x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = x1[x1.index[x1['timestamp']==timestmp]]#.reset_index(drop = True)
                            #print(x1)
                            #x2 = x1[x1.index[(x1['timestamp']==timestmp)&(x1['Flow ID'] == flow)]]#.reset_index(drop = True)
                            #surveys_df[(surveys_df.year >= 1980) & (surveys_df.year <= 1985)]

                            b1 = b1[b1.index[(b1['timestamp']==timestmp)&(b1['Flow ID'] == flow)][0]:].reset_index(drop = True)
                            
                            print(x1)
                            #df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
                            b1 = b1.drop_duplicates().reset_index(drop = True)
                            print(b1)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:] 
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            print('#################################')
                            r'''
                            timestmp = driftfile.read()
                            #print(timestmp)
                            timestmp = np.int64(int(timestmp))
                            #csv = df[:df.index[df['timestamp']==timestmp][0]+1]
                            x1 = x1[x1.index[x1['timestamp']==timestmp][0]:].reset_index(drop = True)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                            '''
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
                ##self.TrainModels(protocol = 'RADIUS', attackLabelsToInt = attackLabelsToInt, input_data=radius_final)
                            #x1 = mqtt_label[mqtt_label.index[mqtt_label['timestamp']==timestmp][0]+1:]
                #print('There are not Data')
        #columns  = list(set(dataframe.columns).intersection(csv.columns))
        
        
        
        
        


        
    
form_data(data_limit = 1000)