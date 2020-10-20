import os
import glob
import sys
import shutil
import subprocess
import csv 
import json
import random
import numpy as np
from statistics import mean, median, variance, stdev
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse

def standardize(datasTrain,datasTest):
    features=[[] for i in range(24)]
    for i in range (24):
        indexRow = i+2
        for data in datasTrain:
            features[i].append(float(data[indexRow]))
        mean=np.array(features[i]).mean()
        std=np.std(features[i])
        for data in datasTrain:
            data[indexRow]=(float(data[indexRow])-mean)/std
        for data in datasTest:
            data[indexRow]=(float(data[indexRow])-mean)/std

def splitDataset(project, variableDependent, release4test):
    # ensure output folder is exist
    os.makedirs("../../datasets/"+project+"/"+variableDependent+"/"+str(release4test), exist_ok=True)

    # train data
    print("datapath for train")
    pathsTrain= []
    for i in range(1, release4test):
        path = "../../datasets/"+project+"/raw/"+variableDependent+"/"+str(i)+".csv"
        pathsTrain.append(path)
    print(pathsTrain)

    datasTrain=[]
    datasValid=[]
    for pathTrain in pathsTrain:
        with open(pathTrain, encoding="utf_8") as f:
            reader = csv.reader(f)
            datasTrain.extend([row for row in reader if row[11]!="0"]) #at least one commit from the last release
    datasBuggy=[]
    datasNotBuggy=[]
    for data in datasTrain:
        if(int(data[1])==1):
            datasBuggy.append(data)
        elif(int(data[1])==0):
            datasNotBuggy.append(data)
    random.seed(0)
    random.shuffle(datasBuggy)
    random.shuffle(datasNotBuggy)

    for i in range(5):
        datasTrain=[]
        datasValid=[]
        validBuggy = copy.deepcopy(datasBuggy[(len(datasBuggy)//5)*i:(len(datasBuggy)//5)*(i+1)])
        validNotBuggy = copy.deepcopy(datasNotBuggy[(len(datasNotBuggy)//5)*i:(len(datasNotBuggy)//5)*(i+1)])
        validBuggy = random.choices(validBuggy, k=len(validNotBuggy))
        datasValid.extend(validBuggy)
        datasValid.extend(validNotBuggy)
        trainBuggy = copy.deepcopy(datasBuggy[:(len(datasBuggy)//5)*i]+datasBuggy[(len(datasBuggy)//5)*(i+1):])
        trainNotBuggy = copy.deepcopy(datasNotBuggy[:(len(datasNotBuggy)//5)*i]+datasNotBuggy[(len(datasNotBuggy)//5)*(i+1):])
        trainBuggy = random.choices(trainBuggy, k=len(trainNotBuggy))
        datasTrain.extend(trainBuggy)
        datasTrain.extend(trainNotBuggy)
        random.shuffle(datasTrain)#最初に1, 次に0ばっかり並んでしまっている。
        random.shuffle(datasValid)#最初に1, 次に0ばっかり並んでしまっている。
        with open('../../datasets/'+project+"/"+variableDependent+"/"+str(release4test)+'/valid'+str(i)+'.csv' , 'w', newline="") as streamFileValid:
            writer = csv.writer(streamFileValid)
            writer.writerows(datasValid)
        with open('../../datasets/'+project+"/"+variableDependent+"/"+str(release4test)+'/train'+str(i)+'.csv' , 'w', newline="") as streamFileTrain:
            writer = csv.writer(streamFileTrain)
            writer.writerows(datasTrain)

    # test data
    print("datapath for test")
    pathTest="../../datasets/"+project+"/raw/"+variableDependent+"/"+str(release4test)+".csv"
    print(pathTest)

    datasTest=[]
    with open(pathTest, encoding="utf_8") as f:
        reader = csv.reader(f)
        datasTest.extend([row for row in reader if row[11]!="0"])
    with open('../../datasets/'+project+"/"+variableDependent+"/"+str(release4test)+'/test.csv' , 'w', newline="") as streamFileTest:
        writer = csv.writer(streamFileTest)
        writer.writerows(datasTest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str,)
    parser.add_argument('--release4test', type=int)
    parser.add_argument('--variableDependent', type=str)
    args = parser.parse_args()

    project = args.project
    release4test = args.release4test
    variableDependent = args.variableDependent

    splitDataset(project, variableDependent, release4test)

if __name__ == '__main__':
    main()