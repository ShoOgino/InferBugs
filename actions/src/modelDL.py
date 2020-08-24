import keras
from keras.optimizers import Adam
from numpy import mean
from numpy import std
from numpy import dstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM,GRU,BatchNormalization,Activation
from keras.layers import Masking
from keras.utils import to_categorical
import os
import glob
import csv
import numpy as np
import random
import csv
import json
import matplotlib.pyplot as plt


dirModel="../../models"
dirDataset="../../datasets"
dirResults="../../results"

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
        print(dataTest[1].shape)
        return dataTest[1]

#class TestCallback(keras.callbacks.Callback):
#    def __init__(self, xValid, yValid,batchSize,resultsValid,indexCV):
#        self.lossBest=1000000
#        self.xValid = xValid
#        self.yValid = yValid
#        self.resultsValid=resultsValid
#        self.batchSize=batchSize
#        self.indexCV=indexCV
#    def on_epoch_end(self, epoch, logs={}):
#        xs, ys = self.xValid, self.yValid
#        loss, mae = self.model.evaluate(xs, ys,batch_size=self.batchSize,verbose=0)
#        ysPredicted=self.model.predict(xs)
#        count=0
#        for i in range(len(ys)):
#            if abs(ys[i]-ysPredicted[i])<0.5:
#                count=count+1
#        acc=count/len(xs)
#
#        self.resultsValid["loss"].append(loss)
#        self.resultsValid["mae"].append(mae)
#        self.resultsValid["acc"].append(acc)
#        print('Validation loss: {}, mae: {}, acc: {}'.format(loss, mae, acc))
#        if loss<self.lossBest:
#            print("lossBest!: {}".format(loss))
#            self.model.save(os.path.join(dirModel,nameProject,str(loss)+'.h5'), #include_optimizer=False)
#            self.lossBest=loss


# fit and evaluate a model
def build(xTrain, yTrain, xValid, yValid, indexCV=0):
    resultsValid={"loss":[],"mae":[], "acc":[],"detail":[[]]}
    verbose, epochs, batch_size = 2, 1000, 128
    n_features, n_outputs = xTrain.shape[1], 1

    model = Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs,activation='sigmoid'))

    adam=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    model.build((None,n_features))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    model.summary()

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')
    history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xValid, yValid))#callbacks=[earlyStopping])#,callbacks=[TestCallback(xValid, yValid, batch_size, resultsValid, indexCV)])
    model.save(os.path.join(dirModel,nameProject,'latest.h5'), include_optimizer=False)

    compare_TV(history)

    return model

def compare_TV(history):

    # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accracy Plt
    fig = plt.figure()
    plt.ylim(0, 2)
    plt.plot(epochs, loss, 'b' ,label = 'lossTrain')
    plt.plot(epochs, acc, 'bo' ,label = 'accTrain')
    plt.title('Training and Validation acc')
    plt.legend()

    plt.plot(epochs, val_loss, 'r' , label= 'lossVal')
    plt.plot(epochs, val_acc, 'ro' , label= 'accVal')
    plt.title('Training and Validation loss')
    plt.legend()

    fig.savefig("result.png")

def predict(nameProject, nameModel, xTest):
    model = keras.models.load_model(os.path.join(dirModel,nameProject,nameModel))
    for x in xTest:
        model.predict(x)


def test(nameProject, nameModel, xTest, yTest):
    model = keras.models.load_model(os.path.join(dirModel,nameProject,nameModel))
    count=0
    ysPredicted=model.predict(xTest)
    tp=0
    fp=0
    fn=0
    tn=0
    for i, yPredicted in enumerate(ysPredicted):
        if(0.5<=yPredicted):
            if(yTest[i]==1):
                tp+=1
            elif(yTest[i]==0):
                fp+=1
        elif(yPredicted<0.5):
            if(yTest[i]==1):
                fn+=1
            elif(yTest[i]==0):
                tn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    fValue=(2*(recall*precision))/(recall+precision)
    acc=(tp+tn)/(tp+fp+tn+fn)
    print(str(tp)+", "+str(fp)+", "+str(fn)+", "+str(tn))
    print("acc: "+ str((tp+tn)/(tp+fp+tn+fn)))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("F-value: "+ str(fValue))



def run(purpose="build", nameProject="", nameModel="", indexCV=2):
    if purpose=="build":
        xTrain, yTrain, xValid, yValid = load_dataset(purpose, nameProject, indexCV)
        model=build(xTrain, yTrain, xValid, yValid, indexCV)
        xTest, yTest=load_dataset("test", nameProject, indexCV)
        test(nameProject, nameModel, xTest, yTest)
    elif purpose=="test":
        xTest, yTest=load_dataset(purpose, nameProject, indexCV)
        test(nameProject, nameModel, xTest, yTest)
    elif purpose=="predict":
        xTest, yTest = load_dataset(purpose, nameProject, indexCV)
        predict(nameModel, xTest)


if __name__ == '__main__':
    purpose="build"
    #purpose="test"
    nameModel="latest.h5"
    #nameModel="0.1734926402568817.h5"
    nameProject="cassandra"
    run(purpose, nameProject, nameModel, 0)