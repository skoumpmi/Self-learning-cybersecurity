import pymongo
import time
import configparser
import pandas as pd
import scipy.spatial.distance as distance
import math 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
#print('============================================')
db = myclient["labelled_netflows_db"]
class estimate_concept_drift:
    def estimate_distance(old_data, new_data, protocol):
        dataframe = new_data.copy()#pd.DataFrame(list(new_data.find()))
        df = dataframe.copy()
        df = df.iloc[:,1:]
        timestamp = df.iloc[-1]['timestamp']
        timestamp_dict = {"timestamp":timestamp}
        df_index = df.index[df['timestamp']==timestamp][0]
        print(df_index)
        if df.iloc[df_index +1:].empty:
                return -1
                r'''
                db.last_drift_point.drop()
                try:
                    db["last_drift_point"].insert_one(timestamp_dict)
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
        
                    db["last_drift_point"].insert_one(n)
                columns  = list(set(dataframe.columns).intersection(old_data.columns))
                new_dataframe = dataframe[columns]
                if len(new_dataframe)<= len(old_data):
                    #csv = csv.iloc[-len(new_dataframe):]
                    csv = old_data.iloc[-len(new_dataframe):]
                else:
                    new_dataframe = new_dataframe.iloc[-len(old_data):]
                print(csv)
                print(new_dataframe)
                df = pd.concat([csv, new_dataframe]).reset_index(drop=True)
                df.replace(attackLabelsToInt, inplace = True)
        
                df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
                'Dst Port'])
        
                #df['DataFrame Column'] = pd.to_numeric(df['machine'])
        
                df.to_csv('mqtt_new_df.csv')
        
                df = df.iloc[:,1:]
                df_cols = df.columns
                print('this is df')
                print(df)
                scaler = MinMaxScaler()
                scaler.fit(df)
                df = pd.DataFrame(scaler.transform(df),columns = df_cols)
                print(df)
                print(math.exp(0.2))
                #breakpoint()
                #print(df['machine'])
                integrated_list = []
                for i in range(0, len(df)):
                    median_list = []
                    sum_elements = 0
                    #print(df.iloc[i].values.flatten())
                    print('ITEM IS {}'.format(i))
                    print('++++++++++++++++++++++++++++++++++++++++++++++')
                    for j in range (0, len(df)):
                        print('==============================================')
                        print(format(0.1, '.17f'))
                        print (format((math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))))
                        print (((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                
                        median_list.append(math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                        sum_elements+=math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                        print(median_list)
                        #print(np.linalg.norm(df.iloc[i].values.flatten()
                        #-df.iloc[j].values.flatten()))
                        #print (((((distance.euclidean
                        #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()))))))
                        #print(2*pow(0.1,2))
                        print('==============================================')
                    median_list = [item / sum_elements for item in median_list]
                    integrated_list.append(median_list)
                print(median_list)
                print(integrated_list)
                print(len(integrated_list))
                old_list =  [0] * int((len(integrated_list)/2))
                new_list =  [0] * int((len(integrated_list)/2))
                for i in range(0, int((len(integrated_list)/2)-1)):
                    #for j in range (0, len(integrated_list[i])):
                    print(integrated_list[i])
                    old_list[i]= sum(integrated_list[i])
                for k in range(int(len(integrated_list)/2), len(integrated_list)):
                    #for l in range (0, len(integrated_list[k])):
                    print(integrated_list[k])
                    new_list[i] = sum(integrated_list[k])
                print(old_list)
                print(new_list) 
                old_list = [item/(len(integrated_list)/2) for item in old_list]
                new_list = [item/(len(integrated_list)/2) for item in new_list]
                print(old_list)
                print(new_list) 
                difference = []

                #initialization of result list

                zip_object = zip(old_list, new_list)

                for old_list_i, new_list_i in zip_object:

                    difference.append(abs(old_list_i-new_list_i))

                print(difference)

                TVD = sum(difference)*0.5
                print(TVD)
                return TVD
                '''
                
        else:

                df = df.iloc[df_index +1:]
                print(timestamp_dict)
                db['last_drift_point_{}'.format(protocol.lower())].drop()
                try:
                    db["last_drift_point_{}".format(protocol.lower())].insert_one(timestamp_dict)
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

            
                    db["last_drift_point_{}".format(protocol.lower())].insert_one(n)


                columns  = list(set(dataframe.columns).intersection(old_data.columns))
                new_dataframe = dataframe[columns]
                if len(new_dataframe)<= len(old_data):
                    #csv = csv.iloc[-len(new_dataframe):]
                    csv = old_data.iloc[-len(new_dataframe):]
                else:
                    new_dataframe = new_dataframe.iloc[-len(old_data):]
                print(csv)
                print(new_dataframe)
                df = pd.concat([csv, new_dataframe]).reset_index(drop=True)
                attack_dict = list(list(db['{}_attacks'.format(protocol.lower())].find()[0].items())[1][1].values())[-1]+1
                attackLabelsToInt = {"Label":attack_dict}
                df.replace(attackLabelsToInt, inplace = True)
        
                df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
                'Dst Port'])
        
                #df['DataFrame Column'] = pd.to_numeric(df['machine'])
        
                df.to_csv('mqtt_new_df.csv')
        
                df = df.iloc[:,1:]
                df_cols = df.columns
                print('this is df')
                print(df)
                scaler = MinMaxScaler()
                scaler.fit(df)
                df = pd.DataFrame(scaler.transform(df),columns = df_cols)
                print(df)
                print(math.exp(0.2))
                #breakpoint()
                #print(df['machine'])
                integrated_list = []
                for i in range(0, len(df)):
                    median_list = []
                    sum_elements = 0
                    #print(df.iloc[i].values.flatten())
                    print('ITEM IS {}'.format(i))
                    print('++++++++++++++++++++++++++++++++++++++++++++++')
                    for j in range (0, len(df)):
                        print('==============================================')
                        print(format(0.1, '.17f'))
                        print (format((math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))))
                        print (((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                
                        median_list.append(math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2)))))
                        sum_elements+=math.exp((-(pow(distance.euclidean
                        (df.iloc[i].values.flatten(),df.iloc[j].values.flatten()),2))/(2*pow(0.1,2))))
                        print(median_list)
                        #print(np.linalg.norm(df.iloc[i].values.flatten()
                        #-df.iloc[j].values.flatten()))
                        #print (((((distance.euclidean
                        #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()))))))
                        #print(2*pow(0.1,2))
                        print('==============================================')
                    median_list = [item / sum_elements for item in median_list]
                    integrated_list.append(median_list)
                print(median_list)
                print(integrated_list)
                print(len(integrated_list))
                old_list =  [0] * int((len(integrated_list)/2))
                new_list =  [0] * int((len(integrated_list)/2))
                for i in range(0, int((len(integrated_list)/2)-1)):
                    #for j in range (0, len(integrated_list[i])):
                    print(integrated_list[i])
                    old_list[i]= sum(integrated_list[i])
                for k in range(int(len(integrated_list)/2), len(integrated_list)):
                    #for l in range (0, len(integrated_list[k])):
                    print(integrated_list[k])
                    new_list[i] = sum(integrated_list[k])
                print(old_list)
                print(new_list) 
                old_list = [item/(len(integrated_list)/2) for item in old_list]
                new_list = [item/(len(integrated_list)/2) for item in new_list]
                print(old_list)
                print(new_list) 
                difference = []

                #initialization of result list

                zip_object = zip(old_list, new_list)

                for old_list_i, new_list_i in zip_object:

                    difference.append(abs(old_list_i-new_list_i))

                print(difference)

                TVD = sum(difference)*0.5
                print(TVD)
                return TVD

    def estimate_permutated_distance(old_data, new_data):
        ##new_dataframe = pd.DataFrame()
        #print(mqtt)
        #print(csv)
        #for item in mqtt.find():
        dataframe = pd.DataFrame(list(mqtt.find()))
        columns  = list(set(dataframe.columns).intersection(old_data.columns))
        new_dataframe = dataframe[columns]
        print('*****************************************')
        print(new_dataframe)
        print('*****************************************')
        ##new_dataframe = pd.DataFrame()
        #print(mqtt)
        #print(csv)
        #for item in mqtt.find():
        ##for item in new_data.find():
            #print(item)
            ##dataframe = pd.DataFrame.from_dict(item, orient="index").transpose()#.to_frame()
            #dataframe.replace(attackLabelsToInt, inplace = True)
            ##columns  = list(set(dataframe.columns).intersection(old_data.columns))
            ##dataframe = dataframe[columns]
            ##new_dataframe = pd.concat([new_dataframe, dataframe]).reset_index(drop=True)
        #print(new_dataframe)
        if len(new_dataframe)<= len(old_data):
            #csv = csv.iloc[-len(new_dataframe):]
            csv = old_data.iloc[-len(new_dataframe):]
        else:
            new_dataframe = new_dataframe.iloc[-len(old_data):]
        #print(csv)
        #print(new_dataframe)
        df = pd.concat([csv, new_dataframe]).reset_index(drop=True)
        df.replace(attackLabelsToInt, inplace = True)
        df = df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Src Port',
        'Dst Port'])
        #df['DataFrame Column'] = pd.to_numeric(df['machine'])
        
        df = df.iloc[:,1:]
        df_cols = df.columns
        print('this is df')
        print(df)
        #df.to_csv('mqtt_new_df.csv')
        #print(df)
        scaler = MinMaxScaler()
        scaler.fit(df)
        df = pd.DataFrame(scaler.transform(df),columns = df_cols)
        print(df)
        df1 = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
        print(df1)
        df = df1.copy()
        #print(df)
        #print(math.exp(0.2))
        #breakpoint()
        #print(df['machine'])
        integrated_list = []
        for i in range(0, len(df)):
            median_list = []
            sum_elements = 0
            #print(df.iloc[i].values.flatten())
            #print('ITEM IS {}'.format(i))
            #print('++++++++++++++++++++++++++++++++++++++++++++++')
            for j in range (0, len(df)):
                #print('==============================================')
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
                #print(np.linalg.norm(df.iloc[i].values.flatten()
                #-df.iloc[j].values.flatten()))
                #print (((((distance.euclidean
                #(df.iloc[i].values.flatten(),df.iloc[j].values.flatten()))))))
                #print(2*pow(0.1,2))
                #print('==============================================')
            median_list = [item / sum_elements for item in median_list]
            integrated_list.append(median_list)
        #print(median_list)
        #print(integrated_list)
        #print(len(integrated_list))
        old_list =  [0] * int((len(integrated_list)/2))
        new_list =  [0] * int((len(integrated_list)/2))
        for i in range(0, int((len(integrated_list)/2)-1)):
            #for j in range (0, len(integrated_list[i])):
            #print(integrated_list[i])
            old_list[i]= sum(integrated_list[i])
        for k in range(int(len(integrated_list)/2), len(integrated_list)):
            #for l in range (0, len(integrated_list[k])):
            #print(integrated_list[k])
            new_list[i] = sum(integrated_list[k])
        #print(old_list)
        #print(new_list) 
        old_list = [item/(len(integrated_list)/2) for item in old_list]
        new_list = [item/(len(integrated_list)/2) for item in new_list]
        #print(old_list)
        #print(new_list) 
        difference = []

        #initialization of result list

        zip_object = zip(old_list, new_list)

        for old_list_i, new_list_i in zip_object:

            difference.append(abs(old_list_i-new_list_i))

        #print(difference)

        TVD = sum(difference)*0.5
        print(TVD)
        return TVD







r'''
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
#print('============================================')
db = myclient["labelled_netflows_db"]
mqtt = db["labelled_mqtt"]

csv = pd.read_csv("datasets/mqtt_dataset_17-19_03_2020.csv")
attackLabelsToInt = {"Label": {"Normal": 0, "Connection Overflow": 1, "Large Payload": 2, "Unauthorized Subscribe": 3}}
#csv.replace(attackLabelsToInt, inplace = True)
#print(csv)

#estimate_distance(old_data, new_data)
distance = estimate_distance(csv, mqtt)
if distance != -1:
    permutations = 100
    perm_list = []
    for i in range(0, permutations):
        perm_list.append(estimate_permutated_distance(csv, mqtt))
    print(perm_list)
    critical = (np.percentile(perm_list, 95))
    print(distance)
    print(critical)

    if distance > critical:
        print("Concept drift detection") 
    else:
        print("No concept drift detection")
else:
    print('There are not new data')

'''
