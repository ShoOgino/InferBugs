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

losssTrain=[]
losssValid=[]

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 16),
        'max_depth': trial.suggest_int('max_depth', 2,  16),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 16),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'random_state':trial.suggest_int('random_state', 0, 16)
    }
    model = RandomForestClassifier(**params)
    lt=[]
    lv=[]
    for indexCV in range(5):
        xTrain, yTrain, xValid, yValid = load_dataset("build", "cassandra", indexCV)
        model.fit(xTrain, yTrain)
        lossTrain = mean_squared_error(yTrain, model.predict(xTrain))
        lossValid = mean_squared_error(yValid, model.predict(xValid))
        lt.append(lossTrain)
        lv.append(lossValid)
    losssTrain.append(mean(lt))
    losssValid.append(mean(lv))
    return mean(lv)

# load the dataset, returns train and test x and y elements
def load_dataset(purpose="", nameProject='', indexCV=0):
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
            #print(dataTrain[1].shape)
            #print(dataTrain[2].shape)
        with open(os.path.join(dirDataset,nameProject,"valid"+str(indexCV)+".csv")) as f:
            valid = csv.reader(f)
            for i,row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
            dataValid[1]=np.array(dataValid[1])
            dataValid[2]=np.array(dataValid[2])
            #print(dataValid[1].shape)
            #print(dataValid[2].shape)
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
        #print(dataTest[1].shape)
        #print(dataTest[2].shape)
        return dataTest[1],dataTest[2]
    elif purpose=="predict":
        dataTest=[[],[]]
        with open(os.path.join(dirDataset,nameProject,"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
        dataTest[1]=np.array(dataTest[1])
        print(dataTest[1].shape)
        return dataTest[1]

def testModel(model, xTest, yTest):
    accuracyTest=accuracy_score(yTest, model.predict(xTest))
    print("accuracyTest: "+str(accuracyTest))

def test():
    xTrain, yTrain, xValid, yValid=load_dataset("build", "cassandra", 0)
    xTrainTmp=np.delete(xTrain, [3, 9, 11, 12, 19, 23], 1)
    xValidTmp=np.delete(xValid, [3, 9, 11, 12, 19, 23], 1)
    parametersBest, accuracyBest=getParametersHyperTrainingBest(xTrainTmp, yTrain, xValidTmp, yValid)
    model=RandomForestClassifier(
        n_estimators=parametersBest["n_estimators"],
        max_depth=parametersBest["max_depth"],
        min_samples_leaf=parametersBest["min_samples_leaf"],
        min_samples_split=parametersBest["min_samples_split"],
        random_state=parametersBest["random_state"]
    )
    model.fit(np.concatenate([xTrainTmp,xValidTmp]), np.concatenate([yTrain, yValid]))

    purpose="test"
    nameProject="cassandra"
    xTest, yTest=load_dataset(purpose, nameProject)
    xTestTmp=np.delete(xTest, [3, 9, 11, 12, 19, 23], 1)

    ysPredicted=model.predict(xTestTmp)
    tp=0
    fp=0
    fn=0
    tn=0
    count=0
    for i, yPredicted in enumerate(ysPredicted):
        if(0.5<=yPredicted):
            count=count+1
            if(yTest[i]==1):
                tp+=1
            elif(yTest[i]==0):
                fp+=1
        elif(yPredicted<0.5):
            if(yTest[i]==1):
                fn+=1
            elif(yTest[i]==0):
                tn+=1
    print(count)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    fValue=(2*(recall*precision))/(recall+precision)
    acc=(tp+tn)/(tp+fp+tn+fn)
    print(str(tp)+", "+str(fp)+", "+str(fn)+", "+str(tn))
    print("acc: "+ str((tp+tn)/(tp+fp+tn+fn)))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("F-value: "+ str(fValue))
    cm=confusion_matrix(ysPredicted, yTest)
    sns.heatmap(cm, annot=True, fmt="1.0f",  cmap='Blues')
    plt.xlabel("label")
    plt.ylabel("prediction")
    plt.tight_layout()
    plt.savefig('matricsConfusion.png')

def visualizeResult():
    fig = plt.figure()
    #plt.ylim(0, 1)
    plt.plot
    plt.plot(losssTrain, 'b' ,label = 'losssTrain')
    plt.plot(losssValid, 'r' ,label = 'losssValid')
    plt.title('val loss')
    plt.legend()

    fig.savefig("result.png")

def getModelBest(xTrain, yTrain, xValid, yValid):
    datasetBest, parametersBest = getParametersHyperBest(xTrain, yTrain, xValid, yValid)
    model=RandomForestClassifier(
            n_estimators=parametersBest["n_estimators"],
            max_depth=parametersBest["max_depth"],
            min_samples_leaf=parametersBest["min_samples_leaf"],
            min_samples_split=parametersBest["min_samples_split"],
            random_state=parametersBest["random_state"]
    )
    model.fit(datasetBest["xTrain"]+datasetBest["xValid"], datasetBest["yTrain"]+datasetBest["yValid"])
    return model

def getParametersHyperBest(xTrain, yTrain, xValid, yValid):
    features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    accuracies=[]
    features2Drop=[]
    for i in range(len(features)):
        print(str(i)+"/"+str(24))
        accuracyBest=-1
        featureWorst=""
        for feature in features:
            print(feature)
            xTrainTmp=np.delete(xTrain, features2Drop+[feature], 1)
            xValidTmp=np.delete(xValid, features2Drop+[feature], 1)
            parametersBest, accuracy=getParametersHyperTrainingBest(xTrainTmp, yTrain, xValidTmp, yValid)
            if(accuracyBest<accuracy):
                accuracyBest = accuracy
                featureWorst = feature
        print("featureWorst: "+str(featureWorst))
        accuracies.append(accuracyBest)
        features2Drop.append(featureWorst)
        features.remove(featureWorst)
        print(accuracies)
        print(features2Drop)
        print("--------------------")
    max_idx=accuracies.index(max(accuracies))
    xTrainTmp=np.delete(xTrain, features2Drop, 1)
    yTrainTmp=yTrain
    xValidTmp=np.delete(xValid, features2Drop, 1)
    yValidTmp=yValid
    parametersBest, accuracy = getParametersHyperTrainingBest(xTrainTmp, yTrainTmp, xValidTmp, yValidTmp)
    return {"xTrain":xTrainTmp, "yTrain": yTrainTmp, "xValid": xValidTmp, "yValid": yValidTmp}, parametersBest

def getParametersHyperTrainingBest(xTrain, yTrain, xValid, yValid):
    accuracyBest=0
    parametersBest={
        "n_estimators":0,
        "random_state":0,
        "max_depth":0,
        "min_samples_leaf":0,
        "min_samples_split":0
    }
    parameters2Tune = {#5, 3, 2, 2, 0
        'n_estimators'     :[2, 3, 5, 10],
        'max_depth'        :[2, 3, 5, 10],
        'min_samples_leaf' :[2, 3, 5, 10],
        'min_samples_split':[2, 3, 5, 10],
        'random_state'     :[0, 7, 10]
    }
    for n_estimators in parameters2Tune["n_estimators"]:
        for max_depth in parameters2Tune["max_depth"]:
            for min_samples_leaf in parameters2Tune["min_samples_leaf"]:
                for min_samples_split in parameters2Tune["min_samples_split"]:
                    for random_state in parameters2Tune["random_state"]:
                        #print("n_estimators: "+str(n_estimators))
                        #print("max_depth: "+str(max_depth))
                        #print("min_samples_leaf: "+str(min_samples_leaf))
                        #print("min_samples_split: "+str(min_samples_split))
                        #print("random_state: "+str(random_state))
                        accuraciesTrain=[]
                        accuraciesValid=[]
                        model=RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                random_state=random_state
                        )
                        model.fit(xTrain, yTrain)
                        accuraciesTrain.append(model.score(xTrain, yTrain))
                        accuraciesValid.append(accuracy_score(yValid, model.predict(xValid)))

                        accuracyTrainAverage=sum(accuraciesTrain)/len(accuraciesTrain)
                        accuracyValidAverage=sum(accuraciesValid)/len(accuraciesValid)
                        #print("averageTrain:"+str(accuracyTrainAverage))
                        #print("averageValid:"+str(accuracyValidAverage))
                        if(accuracyBest<accuracyValidAverage):
                            accuracyBest=accuracyValidAverage
                            parametersBest={
                                "n_estimators":n_estimators ,
                                "random_state":random_state,
                                "max_depth":max_depth,
                                "min_samples_leaf":min_samples_leaf,
                                "min_samples_split":min_samples_split
                            }
    #print(accuracyBest)
    #print(parametersBest)
    return parametersBest, accuracyBest

def main():
    study = optuna.create_study()
    study.optimize(objective, timeout=60)
    print('params:', study.best_params)
    visualizeResult()

if __name__ == '__main__':
    main()
