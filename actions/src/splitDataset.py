import os
import glob
import sys
import shutil
import subprocess
import csv 
import json
import random
import numpy as np
from statistics import mean, median,variance,stdev
import matplotlib.pyplot as plt
import numpy as np
import copy

def standardize(datasTrain,datasTest):
    features=[[] for i in range(24)]
    for i in range (24):
        indexRow=i+2
        for data in datasTrain:
            features[i].append(float(data[indexRow]))
        mean=np.array(features[i]).mean()
        std=np.std(features[i])
        for data in datasTrain:
            data[indexRow]=(float(data[indexRow])-mean)/std
        for data in datasTest:
            data[indexRow]=(float(data[indexRow])-mean)/std

pathsTrain= [
    "../../datasets/cassandra/hasBeenBuggy/1.csv",
    "../../datasets/cassandra/hasBeenBuggy/2.csv"
    ]
pathTest="../../datasets/cassandra/hasBeenBuggy/3.csv"
datasTrain=[]
datasValid=[]
datasTest=[]
for pathTrain in pathsTrain:
    with open(pathTrain, encoding="utf_8") as f:
        reader = csv.reader(f)
        datasTrain.extend([row for row in reader])
with open(pathTest, encoding="utf_8") as f:
    reader = csv.reader(f)
    datasTest.extend([row for row in reader])
datasTest=random.sample(datasTest, len(datasTest))
# kazu
#datasTestBuggy=[]
#datasTestNotBuggy=[]
#for data in datasTest:
#    if(int(data[1])==1):
#        datasTestBuggy.append(data)
#    else:
#        datasTestNotBuggy.append(data)
#datasTest=[]
#datasTest.extend(datasTestBuggy)
#datasTest.extend(random.sample(datasTestNotBuggy, len(datasTestBuggy)))
standardize(datasTrain, datasTest)
with open('../../datasets/cassandra/test.csv' , 'w', newline="") as streamFileTest:
    writer = csv.writer(streamFileTest)
    writer.writerows(datasTest)


datasBuggy=[]
datasNotBuggy=[]
for data in datasTrain:
    if(int(data[1])==1):
        datasBuggy.append(data)
    elif(int(data[1])==0):
        datasNotBuggy.append(data)
#datasNotBuggy=random.sample(datasNotBuggy, len(datasBuggy))
random.seed(0)

random.shuffle(datasBuggy)
random.shuffle(datasNotBuggy)

for i in range(5):
    datasTrain=[]
    datasValid=[]
    datasValid.extend(copy.deepcopy(datasBuggy[(len(datasBuggy)//5)*i:(len(datasBuggy)//5)*(i+1)]))
    datasValid.extend(copy.deepcopy(datasNotBuggy[(len(datasNotBuggy)//5)*i:(len(datasNotBuggy)//5)*(i+1)]))
    datasTrain.extend(copy.deepcopy(datasBuggy[:(len(datasBuggy)//5)*i]+datasBuggy[(len(datasBuggy)//5)*(i+1):]))
    datasTrain.extend(copy.deepcopy(datasNotBuggy[:(len(datasNotBuggy)//5)*i]+datasNotBuggy[(len(datasNotBuggy)//5)*(i+1):]))
    random.shuffle(datasTrain)
    random.shuffle(datasValid)
    with open('../../datasets/cassandra/valid'+str(i)+'.csv' , 'w', newline="") as streamFileValid:
        writer = csv.writer(streamFileValid)
        writer.writerows(datasValid)
    with open('../../datasets/cassandra/train'+str(i)+'.csv' , 'w', newline="") as streamFileTrain:
        writer = csv.writer(streamFileTrain)
        writer.writerows(datasTrain)
