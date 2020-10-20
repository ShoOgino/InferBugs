import optuna
from keras.layers import Input, concatenate
from keras.layers.core import Activation, Flatten, Reshape, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
import csv
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.keras import TqdmCallback
import sys
import tensorflow as tf
import json
import argparse

dirModel="../../models"
dirDataset="../../datasets"
dirResults="../../results"
project=""
release4test=""
variableDependent = ""
purpose =""


def loadDataset(purpose="", project='', indexCV=0):
    if purpose=="build":
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"train"+str(indexCV)+".csv")) as f:
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
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"valid"+str(indexCV)+".csv")) as f:
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
        dataTrain=[[],[],[]]
        dataValid=[[],[],[]]
        dataTest=[[],[],[]]
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"train"+str(indexCV)+".csv")) as f:
            train = csv.reader(f)
            for i,row in enumerate(train):
                dataTrain[0].append(row[0])
                dataTrain[1].append([float(x) for x in row[2:]])
                dataTrain[2].append(int(row[1]))
        dataTrain[1]=np.array(dataTrain[1])
        dataTrain[2]=np.array(dataTrain[2])
        print(dataTrain[1].shape)
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"valid"+str(indexCV)+".csv")) as f:
            valid = csv.reader(f)
            for i,row in enumerate(valid):
                dataValid[0].append(row[0])
                dataValid[1].append([float(x) for x in row[2:]])
                dataValid[2].append(int(row[1]))
        dataValid[1]=np.array(dataValid[1])
        dataValid[2]=np.array(dataValid[2])
        print(dataValid[1].shape)
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
                dataTest[2].append(int(row[1]))
        dataTest[1]=np.array(dataTest[1])
        dataTest[2]=np.array(dataTest[2])
        print(dataTest[1].shape)
        return np.concatenate([dataTrain[1], dataValid[1]]), np.concatenate([dataTrain[2], dataValid[2]], 0), dataTest[1],dataTest[2]
    elif purpose=="predict":
        dataTest=[[],[]]
        with open(os.path.join(dirDataset,project,variableDependent,release4test,"test.csv")) as f:
            test = csv.reader(f)
            for i,row in enumerate(test):
                dataTest[0].append(row[0])
                dataTest[1].append([float(x) for x in row[2:]])
        dataTest[1]=np.array(dataTest[1])
        #print(dataTest[1].shape)
        return dataTest[1]


def create_model(trial):
    n_outputs = 1
    NOLayers = trial.suggest_int('NOlayers', 1, 3)
    model = Sequential()
    for i in range(NOLayers):
        NOUnits=int(trial.suggest_int('NOUnits{}'.format(i), 16, 1024))
        rateDropout = trial.suggest_uniform('rateDropout{}'.format(i), 0.0, 0.5)
        activation = trial.suggest_categorical('activation{}'.format(i), ['hard_sigmoid', 'linear', 'relu', 'sigmoid', 'softplus', 'softsign', 'tanh'])
        model.add(Dense(NOUnits, activation=activation))
        model.add(Dropout(rateDropout))
    model.add(Dense(n_outputs,activation='sigmoid'))
    return model


def create_optimizer(trial):
    nameOptimizer = trial.suggest_categorical('optimizer', ['sgd', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
    if nameOptimizer == "sgd":
        lrSgd = trial.suggest_loguniform('lrSgd', 1e-5, 1e-2)
        momentumSgd = trial.suggest_uniform('momentumSgd', 0.9, 1)
        #decaySgd = trial.suggest_loguniform('decaySgd', 1e-10, 1e-5)
        #opt = keras.optimizers.SGD(lr=lrSgd, momentum=momentumSgd, decay=decaySgd)
        opt = keras.optimizers.SGD(lr=lrSgd, momentum=momentumSgd)
    elif nameOptimizer == "adagrad":
        #lrAdagrad = trial.suggest_loguniform('lrAdagrad', 1e-5, 1e-2)
        #epsilonAdagrad = trial.suggest_loguniform('epsilonAdagrad', 1e-10, 1e-5)
        #decayAdagrad = trial.suggest_loguniform('decayAdagrad', 1e-10, 1e-5)
        #opt = keras.optimizers.Adagrad(lr=lrAdagrad, epsilon=epsilonAdagrad, decay=decayAdagrad)
        opt = keras.optimizers.Adagrad()
    elif nameOptimizer == "adadelta":
        #lrAdadelta = trial.suggest_loguniform('lrAdadelta', 1e-1, 1e-0)
        #rhoAdadelta = trial.suggest_uniform('rhoAdadelta', 0.95, 1)
        #epsilonAdadelta = trial.suggest_loguniform('epsilonAdadelta', 1e-10, 1e-5)
        #decayAdadelta = trial.suggest_loguniform('decayAdadelta', 1e-10, 1e-5)
        #opt = keras.optimizers.Adadelta(lr=lrAdadelta, rho=rhoAdadelta, epsilon=epsilonAdadelta, decay=decayAdadelta)
        opt = keras.optimizers.Adadelta()
    elif nameOptimizer == 'adam':
        lrAdam = trial.suggest_loguniform('lrAdam', 1e-5, 1e-2)
        beta_1Adam = trial.suggest_uniform('beta1Adam', 0.9, 1)
        beta_2Adam = trial.suggest_uniform('beta2Adam', 0.999, 1)
        epsilonAdam = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-5)
        #decayAdam = trial.suggest_loguniform('decayAdam', 1e-10, 1e-5)
        #opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam, decay=decayAdam, amsgrad=False)
        opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
    elif nameOptimizer == "adamax":
        lrAdamax = trial.suggest_loguniform('lrAdamax', 1e-5, 1e-2)
        beta_1Adamax = trial.suggest_uniform('beta1Adamax', 0.9, 1)
        beta_2Adamax = trial.suggest_uniform('beta2Adamax', 0.999, 1)
        epsilonAdamax = trial.suggest_loguniform('epsilonAdamax', 1e-10, 1e-5)
        #decayAdamax = trial.suggest_loguniform('decayAdamax', 1e-10, 1e-5)
        #opt = keras.optimizers.Adamax(lr=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax, decay=decayAdamax)
        opt = keras.optimizers.Adamax(lr=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax)
    elif nameOptimizer == "nadam":
        lrNadam = trial.suggest_loguniform('lrNadam', 1e-5, 2e-3)
        beta_1Nadam = trial.suggest_uniform('beta1Nadam', 0.9, 1)
        beta_2Nadam = trial.suggest_uniform('beta2Nadam', 0.999, 1)
        #epsilonNadam = trial.suggest_loguniform('epsilonNadam', 1e-10, 1e-5)
        opt = keras.optimizers.Nadam(lr=lrNadam, beta_1=beta_1Nadam, beta_2=beta_2Nadam)
    return opt

def analyzeResult(ysPredicted, yLabel):
    tp=0
    fp=0
    fn=0
    tn=0
    for i, yPredicted in enumerate(ysPredicted):
        if(0.5<=yPredicted):
            if(yLabel[i]==1):
                tp+=1
            elif(yLabel[i]==0):
                fp+=1
        elif(yPredicted<0.5):
            if(yLabel[i]==1):
                fn+=1
            elif(yLabel[i]==0):
                tn+=1
    precision=tp/(tp+fp+0.1)
    recall=tp/(tp+fn+0.1)
    fValue=(2*(recall*precision))/(recall+precision+0.1)
    acc=(tp+tn)/(tp+fp+tn+fn+0.1)
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


def saveGraphTrial(history):
    # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig = plt.figure()
    plt.ylim(0, 2)
    plt.plot(epochs, loss, 'b' ,label = 'lossTrain')
    plt.plot(epochs, acc, 'r' ,label = 'accTrain')
    plt.plot(epochs, val_loss, 'bo' , label= 'lossVal')
    plt.plot(epochs, val_acc, 'ro' , label= 'accVal')
    plt.title('result')
    plt.legend()

    pathLogGraph = dirResults + "/" \
        + project + "_" \
        + variableDependent + "_" \
        + release4test + "_" \
        + model + "_" \
        + dateStart + "_" \
        + trial.number \
        + '.png'
    fig.savefig(pathLogGraph)
    plt.clf()
    plt.close()

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
            self.model.save(os.path.join(dirModel,project,str(loss)+'.h5'), include_optimizer=False)
            self.lossBest=loss
            ysPredictedLabel=[]
            for yPredicted in ysPredicted:
                if yPredicted<0.5:
                    ysPredictedLabel.append(0)
                else:
                    ysPredictedLabel.append(1)
            #print(ysPredictedLabel)
            analyzeResult(ysPredictedLabel, ys)


def objective(trial):
    resultsValid={"loss":[],"mae":[], "acc":[],"detail":[[]]}
    xTrain, yTrain, xValid, yValid = loadDataset("build", project)

    verbose, epochs, sizeBatch = 0, 1000, trial.suggest_int("sizeBatch", 32, 256)
    n_features, n_outputs = xTrain.shape[1], 1

    model = create_model(trial)
    opt = create_optimizer(trial)

    model.build((None,n_features))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xValid, yValid),     callbacks=[EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')])#callbacks=[TestCallback(xValid, yValid,     batch_size, resultsValid, 0)])
    saveGraphTrial(history)


    # 最小値のエポック数+6までの値の平均を取る。
    lossesVal = history.history['val_loss']
    lossValMin = min(lossesVal)
    indexValMin = lossesVal.index(lossValMin)
    indexLast = len(lossesVal)-1
    index5Forward = indexValMin+5 if indexValMin+5 < indexLast else indexLast
    score=0
    for i in range(6):
        score += lossesVal[index5Forward-i]
    score = score / 6

    #全体のログをtxt+グラフ形式で出力
    pathLogOverall = dirResults + "/" \
        + project + "_" \
        + variableDependent + "_" \
        + release4test + "_" \
        + model + "_" \
        + dt_now.strftime('%Y%m%d%H%M%S') + "_" \
        + trial.number \
        + '.txt'
    with open(pathLogOverall, mode='a') as f:
        f.write('\n')
        f.write(score)
        f.write(",")
        f.write(trial.params)

    return score

def test(hp):
    n_outputs = 1
    model = Sequential()
    for i in range(int(hp["NOlayers"])):
        model.add(Dense(hp["NOUnits"+str(i)], activation=hp["activation"+str(i)]))
        model.add(Dropout(hp["rateDropout"+str(i)]))
        model.add(Dense(n_outputs,activation='sigmoid'))

    resultsValid={"loss":[],"mae":[], "acc":[],"detail":[[]]}
    xTrain, yTrain, xTest, yTest = loadDataset("test", project)

    verbose, epochs, sizeBatch = 1, 10, hp["sizeBatch"]
    n_features, n_outputs = xTrain.shape[1], 1

    model.build((None,n_features))
    if hp["optimizer"] == "sgd":
        lrSgd = hp["lrSgd"]
        momentumSgd = hp["momentumSgd"]
        decaySgd = hp["decaySgd"]
        opt = keras.optimizers.SGD(lr=lrSgd, momentum=momentumSgd, decay=decaySgd)
    elif hp["optimizer"] == "adagrad":
        lrAdagrad = hp["lrAdagrad"]
        epsilonAdagrad = hp["epsilonAdagrad"]
        decayAdagrad = hp["decayAdagrad"]
        opt = keras.optimizers.Adagrad(lr=lrAdagrad, epsilon=epsilonAdagrad, decay=decayAdagrad)
    elif hp["optimizer"] == "adadelta":
        lrAdadelta = hp["lrAdadelta"]
        rhoAdadelta = hp["rhoAdadelta"]
        epsilonAdadelta = hp["epsilonAdadelta"]
        decayAdadelta = hp["decayAdadelta"]
        opt = keras.optimizers.Adadelta(lr=lrAdadelta, rho=rhoAdadelta, epsilon=epsilonAdadelta, decay=decayAdadelta)
    elif hp["optimizer"] == 'adam':
        lrAdam = hp["lrAdam"]
        beta_1Adam = hp["beta1Adam"]
        beta_2Adam = hp["beta2Adam"]
        epsilonAdam = hp["epsilonAdam"]
        decayAdam = hp["decayAdam"]
        opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam, decay=decayAdam, amsgrad=False)
    elif hp["optimizer"] == "adamax":
        lrAdamax = hp["lrAdamax"]
        beta_1Adamax = hp["beta1Adamax"]
        beta_2Adamax = hp["beta2Adamax"]
        epsilonAdamax = hp["epsilonAdamax"]
        decayAdamax = hp["decayAdamax"]
        opt = keras.optimizers.Adamax(lr=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax, decay=decayAdamax)
    elif hp["optimizer"] == "nadam":
        lrNadam = hp["lrNadam"]
        beta_1Nadam = hp["beta1Nadam"]
        beta_2Nadam = hp["beta2Nadam"]
        epsilonNadam = hp["epsilonNadam"]
        opt = keras.optimizers.Nadam(lr=lrNadam, beta_1=beta_1Nadam, beta_2=beta_2Nadam, epsilon=epsilonNadam, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xTest, yTest), callbacks=[EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto'), TestCallback(xTest, yTest, sizeBatch, resultsValid, 0)])
    compareTV(history)

def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")    

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str)
    parser.add_argument('--release4test', type=str)
    parser.add_argument('--variableDependent', type=str)
    parser.add_argument('--purpose', type=str)
    parser.add_argument('--modelAlgorithm', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--parameterHyper', type=str)
    args = parser.parse_args()

    global project 
    project = args.project
    global release4test 
    release4test = args.release4test
    global variableDependent 
    variableDependent = args.variableDependent
    global purpose 
    purpose = args.purpose
    modelAlgorithm = args.modelAlgorithm
    parameterHyper = args.parameterHyper
    global model
    model = args.model

    global dateStart
    dateStart=dt_now.strftime('%Y%m%d%H%M%S')

    if purpose=="test":
        json_open = open(os.path.join(dirDataset, project, release4test, 'hpDL.json'), 'r')
        hp = json.load(json_open)
        test(hp)
    elif purpose=="search":
        study = optuna.create_study()
        study.optimize(objective, timeout = 60*60*8)
        pathLogGraphOverall = dirResults + "/" \
            + project + "_" \
            + variableDependent +  "_" \
            + release4test + "_" \
            + model + "_" \
            + dt_now.strftime('%Y%m%d%H%M%S') + "_" \
            + trial.number \
            + '.png'


if __name__ == '__main__':
    main()