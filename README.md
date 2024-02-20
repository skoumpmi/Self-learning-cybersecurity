1. First of all it should be created an anaconda environment with the dependencies are described in requirements.txt
2. Then it should be installed mongodb in machine
3. Initail models per method and protocol should be stored in project from previous runnings
4. It should be run the 'post_endpoint.py' file.
5. The response of file should be "Running on http://localhost:5000/ (Press CTRL+C to quit)" means that flask call a local server.
When there is a http request it should be appeared 'http://something.sth:port/"
6.Data from new attacks should be downloaded in mongo db netflows
7. Open the Postman or with a corresponding way send the request: Content-type:"http://127.0.0.1:5000/GetUserInteraction"
Request:"[ {

"value": "160.40.49.209-160.40.49.226-33747-1883-6" ,
"Label": "Normal"},


{
"value": "160.40.49.166-160.40.49.226-61961-1883-6",
"Label":"Normal"
} 
]"
It means that neflow eg with id=="160.40.49.166-160.40.49.226-61961-1883-6" is already in mongo db netflow data.
6. Labels normal means that deep learning model will be retrained with random sample from initial csvs and historic interaction 
between user and system. In contrast, machine learning models are trained from scratch  
New attack , other than known attacks, means that both deep learning and machine learning models are trained from scratch.


Notifications:
During training process in mongo db will be appeared
- protocol_attacks collection like eg "modbus_attacks" which is a list of known attacks
- history_protocol like "history_modbus" where it will be stored the data points which labels' are changed by user.
- retrain_mqtt is only datapoints from final interaction between user and Self-learning module. In this case it is created 
these collections for possible future usage.


![alt text](https://varlab.iti.gr:9443/H2020-Projects/ICT-IoT-Energy/SPEAR/uploads/2dd76f135b086c93ebd8b9a25dbe3c55/Self-learning.drawio.png?raw=true)
