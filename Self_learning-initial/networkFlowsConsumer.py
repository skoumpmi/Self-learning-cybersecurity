#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 SPEAR Authors in the Centre for Research and Technology Hellas, Information Technology Institute (CERTH-ITI). All rights reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential in the context of the EU H2020 SPEAR project.

import datetime
from kafka import KafkaConsumer
import json
import configparser
import sys
import traceback

#sys.path.append('/home/certh/tmp/allim/src/app/')


class SmartHomeKafkaConsumer_SH():

    def __init__(self):
        # create event dictionary
        self.load_kafka_config()
        try:
            print('Bootstrapserver: ', '[{0}:{1}]'.format(
                self.SmartHomeSIEM_Cert['kafka_ip'], self.SmartHomeSIEM_Cert['kafka_port']))
            self.consumer = KafkaConsumer(group_id='selflearning',
                                          value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                                          bootstrap_servers='[{0}:{1}]'.format(
                                              self.SmartHomeSIEM_Cert['kafka_ip'],
                                              self.SmartHomeSIEM_Cert['kafka_port']),
                                          security_protocol='SSL',
                                          ssl_ciphers='ALL',
                                          ssl_check_hostname=False,
                                          ssl_cafile=self.SmartHomeSIEM_Cert['kafka_ca_file'],
                                          ssl_certfile=self.SmartHomeSIEM_Cert['kafka_cert_file'],
                                          ssl_keyfile=self.SmartHomeSIEM_Cert['kafka_key_file'],
                                          ssl_password=self.SmartHomeSIEM_Cert['kafka_pass'],
                                          max_poll_interval_ms = 5000)
                                          # api_version_auto_timeout_ms= int(self.SmartHomeSIEM_Cert['api_version_auto_timeout_ms']),
                                          #auto_offset_reset='latest')
            # enable_auto_commit=True,
            # auto_commit_interval_ms=2000)
            print("Connected to KAFKA...")
            print(self.SmartHomeSIEM_Cert['kafka_ip'])
        except:
            traceback.print_exc()
            print("SmartHomeKafkaConsumer Object is not created! :(. Check kafka credentials.")
            sys.exit(1)

    def load_kafka_config(self):
        self.config_kafka = configparser.ConfigParser()
        try:
            self.config_kafka.read('config/config_kafka.ini')
            # Load the SPEAR SIEM Kafka certificates for the SmartHome use case.
            self.SmartHomeSIEM_Cert = dict(self.config_kafka.items('SPEAR SIEM TECNALIA KAFKA'))
            # Load SPEAR SIEM Kafka Topics for the SmartHome use case.
            self.SmartHomeSIEM_Topics = dict(self.config_kafka.items('SMART HOME 2 SPEAR KAFKA TOPICS'))
        except:
            traceback.print_exc()
            print("Check the configuration file or the home directory of the executable")
            sys.exit(1)
        # Debug prints - To be removed.
        print("Config Sections: " + str(self.config_kafka.sections()))
        print("The kafka CERT are: " + str(self.SmartHomeSIEM_Cert))
        print("The kafka SmartHome topics are: " + str(self.SmartHomeSIEM_Topics))



