from src.utility import UtilPath
import numpy as np
import os
import csv

class Dataset:
    def __init__(self, project, variableDependent, release4Test, purpose):
        self.project = project
        self.variableDependent = variableDependent
        self.release4Test = release4Test
        self.porpose = purpose

    def getTrain4Search(self, indexCV=0):
        dataTrain=[[],[],[]]
        with open(os.path.join(UtilPath.Datasets(),self.project,self.variableDependent,self.release4Test,"train"+str(indexCV)+".csv")) as f:
            train = csv.reader(f)
            for i,row in enumerate(train):
                #print(item["id"])
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
            dataTrain[1]=np.array(dataTrain[1])
            dataTrain[2]=np.array(dataTrain[2])
            #print(dataTrain[1].shape)
            #print(dataTrain[2].shape)
            return dataTrain[1], dataTrain[2]

    def getValid4Search(self, indexCV=0):
        dataValid=[[],[],[]]
        with open(os.path.join(UtilPath.Datasets(),self.project,self.variableDependent,self.release4Test,"valid"+str(indexCV)+".csv")) as f:
            valid = csv.reader(f)
            for i,row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
            dataValid[1]=np.array(dataValid[1])
            dataValid[2]=np.array(dataValid[2])
            #print(dataValid[1].shape)
            #print(dataValid[2].shape)
        return dataValid[1], dataValid[2]

    def getTrain4Test(self, indexCV=0):
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        with open(os.path.join(UtilPath.Datasets(),self.project,self.variableDependent,self.release4Test,"train"+str(indexCV)+".csv")) as f:
            train = csv.reader(f)
            for i,row in enumerate(train):
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
        dataTrain[1]=np.array(dataTrain[1])
        dataTrain[2]=np.array(dataTrain[2])
        print(dataTrain[1].shape)
        with open(os.path.join(UtilPath.Datasets(),self.project,self.variableDependent,self.release4Test,"valid"+str(indexCV)+".csv")) as f:
            valid = csv.reader(f)
            for i,row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
        dataValid[1]=np.array(dataValid[1])
        dataValid[2]=np.array(dataValid[2])
        print(dataValid[1].shape)
        return np.concatenate([dataTrain[1], dataValid[1]]), np.concatenate([dataTrain[2], dataValid[2]], 0)

    def getTest4Test(self, indexCV=0):
            dataTest=[[],[],[]]
            with open(os.path.join(UtilPath.Datasets(),self.project,self.variableDependent,self.release4Test,"test.csv")) as f:
                test = csv.reader(f)
                for i,row in enumerate(test):
                    dataTest[0].append(row[0])
                    dataTest[1].append([float(x) for x in row[2:]])
                    dataTest[2].append(int(row[1]))
            dataTest[1]=np.array(dataTest[1])
            dataTest[2]=np.array(dataTest[2])
            print(dataTest[1].shape)
            return dataTest[1],dataTest[2]