
import pandas as pd
import pymongo
import numpy as np
import os

class data_creation:

    def __init__(self, protocol,  db):
        self.protocol = protocol
        #self.attackLabelsToInt = attackLabelsToInt
        self.db = db
        

    def create_data(self, attackLabelsToInt):
        # It is concated the data points from csvs and the data points from historic data
        # of historic collection
        # initial csv is read
        #IN THIS CASE THE NAME OF CSVS IS CHANGED AS TO BE OPTIMIZED THE CODE
        ##csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/datasets/{}_dataset.csv".format(self.protocol.lower()))
        csv = pd.read_csv(open(os.getcwd(),"../datasets/{}_dataset.csv".format(self.protocol.lower())))
        # 1. we take into account the historical data from user interactions from the
        # past and now.
        history = self.db["history_{}".format(self.protocol.lower())]
        label = self.db["labelled_{}".format(self.protocol.lower())]
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
        print('++++++++++++++++++++++++++++++')
        print(csv['Label'].unique())
        for column in columns:
            
            if type(csv.iloc[0][column]) == np.float64:
                csv = csv.astype({"{}".format(column): np.float32})
            if type(csv.iloc[0][column]) == np.int64:
                csv = csv.astype({"{}".format(column): np.int32})

        csv.info(memory_usage="deep")
        return csv

    def create_short_data(self,attackLabelsToInt,model_type):
        ##init_csv = pd.read_csv("/home/smarthome/SPEAR/varlab_dev/SPEAR/development/BDAC/Self_learning_retrain/datasets/{}_dataset.csv".format(self.protocol.lower()))
        init_csv = pd.read_csv(os.path.join(os.getcwd(),"../datasets/{}_dataset.csv".format(self.protocol.lower())))
        attack_list = []
        if self.protocol == "MODBUS":
            attackLabelsToIntInit = {
                "Label": {'Normal': 0, 'UID brute force': 1, 'Enumeration Function': 2,
                          'Fuzzing Read Holding Registers': 3}}
        if self.protocol == "MQTT":
            attackLabelsToIntInit = {
                "Label": {"Normal": 0, "Unauthorized Subscribe": 1, "Large Payload": 2, "Connection Overflow": 3}}
        for k in attackLabelsToIntInit['Label']:
            attack_list.append(k)

        counter = 0
        #limit = 128#64  # -len_attacks
        limit = 16384
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
                columns =  [col for col in columns if col != 'Label' ] + ['Label'] 
                new_df = new_df[columns]
                final_df = pd.concat([final_df, new_df])
                
        

        list_indices = final_df.index.values.tolist()
        # We get some other random samples to have at least 64 data points.
        # Otherwise we have bug in retraining process
        other_df = pd.DataFrame()
        while other_df.empty:
            #if len(final_df) < 64:
            if len(final_df) < 16384:
                #limit = 64 - len(final_df)
                limit = 16384 - len(final_df)
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
        print('++++++++++++++++++++++++++++++')
        print(final_df['Label'].unique())
        ##1. we take into account the historical data from user interactions from the
        # past and now.
        # x = pd.DataFrame(list(modbus.find({"Flow ID":netflow_id})))
        #train_db = pd.DataFrame(list(db["history_{}".format(protocol.lower())].find()))
        if "history_{}".format(self.protocol.lower()) in self.db.list_collection_names() and "labelled_{}".format(self.protocol.lower()) in self.db.list_collection_names():
                train_db = pd.concat([pd.DataFrame(list(self.db["history_{}".format(self.protocol.lower())].find())),pd.DataFrame(list(self.db["labelled_{}".format(self.protocol.lower())].find()))]).reset_index(drop = True)
        if "history_{}".format(self.protocol.lower()) in self.db.list_collection_names():
                train_db = pd.DataFrame(list(self.db["history_{}".format(self.protocol.lower())].find()))
        else:
                train_db = pd.DataFrame()
        
        final_columns = list(set(train_db.columns).intersection(final_df.columns))
        final_columns =  [col for col in final_columns if col != 'Label' ] + ['Label'] 
        train_db = train_db[final_columns]
        
        # 4. We get the columns that is the same from initial csvs and the collection historic data.
        final_df = pd.concat([final_df, train_db])
        final_df = final_df.reset_index(drop=True)
        final_df.replace(attackLabelsToInt, inplace=True)
        columns = final_df.columns
        
        for column in columns:
            
            if type(final_df.iloc[0][column]) == np.float64:
                final_df = final_df.astype({"{}".format(column): np.float32})
            if type(final_df.iloc[0][column]) == np.int64:
                final_df = final_df.astype({"{}".format(column): np.int32})
        print('FINAL DF IS:')
        print('=============================')
        print(final_df)
        return final_df

    def convert_to_np(self, collection, record):
        try:
            collection.insert(record)
        except pymongo.errors.InvalidDocument:
            # Python 2.7.10 on Windows and Pymongo are not forgiving
            # If you have foreign data types you have to convert them
            n = {}
            for k, v in record.items():

                if isinstance(v, np.int64):
                    v = int(v)

                n[k] = v

            collection.insert(n)

