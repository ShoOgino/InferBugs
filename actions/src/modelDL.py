from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM,GRU,BatchNormalization,Activation
from keras.layers import Masking
from keras.utils import to_categorical
from matplotlib import pyplot
import argparse
import os
import glob
import csv
import numpy as np
import random
import keras
import matplotlib.pyplot as plt
import csv
import json
csv.field_size_limit(1000000000)
from keras import backend as k
np.set_printoptions(threshold=np.inf)

dirModel="..\\..\\models"
dirDataset="../../datasets"
dirResults="../../results"

# load the dataset, returns train and test x and y elements
def load_dataset(purpose="", nameProject='', release=1, indexCV=0):
    if purpose=="build":
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        with open(os.path.join(dirDataset,nameProject,str(release),"train"+str(indexCV)+".csv")) as f:
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
        with open(os.path.join(dirDataset,nameProject,str(release),"valid"+str(indexCV)+".csv")) as f:
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
        with open(os.path.join(dirDataset,nameProject,str(release+1),"test.csv")) as f:
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
        with open(os.path.join(dirDataset,nameProject,str(release+1),"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
        dataTest[1]=np.array(dataTest[1])
        print(dataTest[1].shape)
        return dataTest[1]

class TestCallback(keras.callbacks.Callback):
    def __init__(self, xValid, yValid,batchSize,resultsValid,indexCV):
        self.lossBest=1000000
        self.xValid = xValid
        self.yValid = yValid
        self.resultsValid=resultsValid
        self.batchSize=batchSize
        self.indexCV=indexCV
    def on_epoch_end(self, epoch, logs={}):
        xs, ys = self.xValid, self.yValid
        loss, mae = self.model.evaluate(xs, ys,batch_size=self.batchSize,verbose=0)
        ysPredicted=self.model.predict(xs)
        count=0
        for i in range(len(ys)):
            if abs(ys[i]-ysPredicted[i])<0.5:
                count=count+1
        acc=count/len(xs)

        self.resultsValid["loss"].append(loss)
        self.resultsValid["mae"].append(mae)
        self.resultsValid["acc"].append(acc)
        print('Validation loss: {}, mae: {}, acc: {}'.format(loss, mae, acc))
        if loss<self.lossBest:
            print("lossBest!: {}".format(loss))
            #self.model.save(str(loss)+'.h5', include_optimizer=False)
            self.lossBest=loss


# fit and evaluate a model
def build(xTrain, yTrain, xValid, yValid, indexCV=0):
    resultsValid={"loss":[],"mae":[], "acc":[],"detail":[[]]}
    verbose, epochs, batch_size = 2, 70, 64
    n_features, n_outputs = xTrain.shape[1], 1

    model = Sequential()
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs,activation='linear'))

    adam=keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.build((None,n_features))
    model.compile(loss='mse', optimizer=adam, metrics=['mean_absolute_error'])
    model.summary()

    history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose,callbacks=[TestCallback(xValid, yValid, batch_size, resultsValid, indexCV)])
    
    model.save(os.path.join(dirModel,nameProject,"latest.h5"), include_optimizer=False)

    return model


def predict(nameProject, nameModel, xTest):
    model = keras.models.load_model(os.path.join(dirModel,nameProject,nameModel), compile=False)
    for x in xTest:
        model.predict(x)


def test(nameProject, nameModel, xTest, yTest):
    model = keras.models.load_model(os.path.join(dirModel,nameProject,nameModel), compile=False)
    count=0
    ysPredicted=model.predict(xTest)
    tp=0
    fp=0
    fn=0
    tn=0
    for i, yPredicted in enumerate(ysPredicted):
        if(0.5<yPredicted):
            if(0.5<yTest[i]):
                tp+=1
            elif(yTest[i]<0.5):
                fp+=1
        else:
            if(0.5<yTest[i]):
                fn+=1
            elif(yTest[i]<0.5):
                tn+=1
    print("precision: " + str(tp/(tp+fp)))
    print("recall: " + str(tp/(tp+fn)))
    print("acc: "+ str((tp+tn)/(tp+fp+tn+fn)))


def run(purpose="build", nameProject="", release=1, nameModel="", indexCV=0):
    if purpose=="build":
        xTrain, yTrain, xValid, yValid = load_dataset(purpose, nameProject, release, indexCV)
        model=build(xTrain, yTrain, xValid, yValid, indexCV)
        xTest, yTest=load_dataset("test", nameProject, release, indexCV)
        test(nameProject, nameModel, xTest, yTest)
    elif purpose=="test":
        xTest, yTest=load_dataset(purpose, nameProject, release, indexCV)
        test(nameProject, nameModel, xTest, yTest)
    elif purpose=="predict":
        xTest, yTest = load_dataset(purpose, nameProject, release, indexCV)
        predict(nameModel, xTest)


if __name__ == '__main__':
    purpose="build"
    nameModel="latest.h5"
    nameProject="cassandra"
    run(purpose, nameProject, 1, nameModel, 0)