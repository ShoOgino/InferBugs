# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from pandas.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV
import os
import csv
import numpy as np
import copy
dirModel="../../models"
dirDataset="../../datasets"
dirResults="../../results"
import seaborn as sns
from sklearn.metrics import confusion_matrix
import optuna
from statistics import mean
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

losssTrain=[]
losssValid=[]
accs=[]
xTrains=[]
yTrains=[]
xValids=[]
yValids=[]

def loadDataset(purpose="", nameProject='', indexCV=0):
    if purpose=="build":
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        with open(os.path.join(dirDataset,nameProject,"train"+str(indexCV)+".csv")) as f:
            train = csv.reader(f)
            for i,row in enumerate(train):
                #print(item["id"])
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
            dataTrain[1]=np.array(dataTrain[1])
            dataTrain[2]=np.array(dataTrain[2])
            print(dataTrain[1].shape)
            print(dataTrain[2].shape)
        with open(os.path.join(dirDataset,nameProject,"valid"+str(indexCV)+".csv")) as f:
            valid = csv.reader(f)
            for i,row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
            dataValid[1]=np.array(dataValid[1])
            dataValid[2]=np.array(dataValid[2])
            print(dataValid[1].shape)
            print(dataValid[2].shape)
        return dataTrain[1], dataTrain[2], dataValid[1], dataValid[2]
    elif purpose=="test":
        dataTest=[[],[],[]]
        with open(os.path.join(dirDataset,nameProject,"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
                dataTest[2].append(int(row[1]))
        dataTest[1]=np.array(dataTest[1])
        dataTest[2]=np.array(dataTest[2])
        print(dataTest[1].shape)
        print(dataTest[2].shape)
        return dataTest[1],dataTest[2]
    elif purpose=="predict":
        dataTest=[[],[]]
        with open(os.path.join(dirDataset,nameProject,"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
        dataTest[1]=np.array(dataTest[1])
        #print(dataTest[1].shape)
        return dataTest[1]

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 2,  256),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2,  256),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 256),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 256),
        'random_state':42}
    model = RandomForestClassifier(**params)
    lt=[]
    lv=[]
    accs=[]
    for indexCV in range(5):
        model.fit(xTrains[indexCV], yTrains[indexCV])
        lossTrain = mean_squared_error(yTrains[indexCV], model.predict(xTrains[indexCV]))
        lossValid = mean_squared_error(yValids[indexCV], model.predict(xValids[indexCV]))
        acc = accuracy_score(yValids[indexCV], model.predict(xValids[indexCV]))
        lt.append(lossTrain)
        lv.append(lossValid)
        accs.append(acc)
    losssTrain.append(mean(lt))
    losssValid.append(mean(lv))
    return 1-mean(accs)

def visualizeResult():
    fig = plt.figure()
    plt.plot
    plt.plot(losssTrain, 'b' ,label = 'losssTrain')
    plt.plot(losssValid, 'r' ,label = 'losssValid')
    plt.title('val loss')
    plt.legend()
    fig.savefig("result.png")

def getParametersHyperBest():
    for indexCV in range(5):
        xTrain, yTrain, xValid, yValid = loadDataset("build", "cassandra", indexCV)
        xTrains.append(xTrain)
        yTrains.append(yTrain)
        xValids.append(xValid)
        yValids.append(yValid)
    study = optuna.create_study()
    study.optimize(objective, timeout=60*60*6)
    print('params:', study.best_params)
    visualizeResult()

def analyzeResult(ysPredicted, yLabel):
    tp=0
    fp=0
    fn=0
    tn=0
    count=0
    for i, yPredicted in enumerate(ysPredicted):
        if(0.5<=yPredicted):
            count=count+1
            if(yLabel[i]==1):
                tp+=1
            elif(yLabel[i]==0):
                fp+=1
        elif(yPredicted<0.5):
            if(yLabel[i]==1):
                fn+=1
            elif(yLabel[i]==0):
                tn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    fValue=(2*(recall*precision))/(recall+precision)
    acc=(tp+tn)/(tp+fp+tn+fn)
    #print(str(tp)+", "+str(fp)+", "+str(fn)+", "+str(tn))
    print("acc: "+ str(acc))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("F-value: "+ str(fValue))
    cm=confusion_matrix(ysPredicted, yLabel)
    sns.heatmap(cm, annot=True, fmt="1.0f",  cmap='Blues')
    plt.xlabel("label")
    plt.ylabel("prediction")
    plt.tight_layout()
    plt.savefig('matricsConfusion.png')

def test(parameters):
    xTrain, yTrain, xValid, yValid=loadDataset("build", "cassandra", 2)
    x=np.concatenate([xTrain, xValid])
    y=np.concatenate([yTrain, yValid])
    for i in range(100):
        model=RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            max_leaf_nodes=parameters["max_leaf_nodes"],
            min_samples_leaf=parameters["min_samples_leaf"],
            min_samples_split=parameters["min_samples_split"],
            random_state=i)
        model.fit(x,y)
        xTest, yTest=loadDataset("test", "cassandra")
        ysPredicted=model.predict(xTest)
        analyzeResult(ysPredicted, yTest)

def main():
    #getParametersHyperBest()
    parameters= {'n_estimators': 232, 'max_depth': 69, 'max_leaf_nodes': 209, 'min_samples_leaf': 9, 'min_samples_split': 63, "random_state":0}
    test(parameters)

if __name__ == '__main__':
    main()
