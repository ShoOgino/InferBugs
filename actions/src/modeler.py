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
from tensorflow.keras.callbacks import ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.keras import TqdmCallback
import sys
import tensorflow as tf
import json
import argparse
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


class Modeler:
    def __init__(self, dirResults):
        self.dirResults = dirResults

    def saveGraphTrain(self, history, numberTrial):
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
        plt.title(str(numberTrial))
        plt.legend()

        pathLogGraph = os.path.join(self.dirResults, str(numberTrial) + '.png')
        fig.savefig(pathLogGraph)
        plt.clf()
        plt.close()

    def search(self, xTrain, yTrain, xValid, yValid, modelAlgorithm):
        def set_objective(xTrain, yTrain, xValid, yValid, modelAlgorithm):
            def objectiveRF(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 2, 256),
                    'max_depth': trial.suggest_int('max_depth', 2,  256),
                    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2,  256),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 256),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 256),
                    'random_state':42}
                model = RandomForestClassifier(**params)
                model.fit(xTrain, yTrain)
                score = mean_squared_error(yValid, model.predict(xValid))
                #全体のログをtxt形式で出力
                with open(os.path.join(self.dirResults, "results.txt"), mode='a') as f:
                    f.write(str(score)+","+str(trial.params)+'\n')
                return score
            def objectiveDNN(trial):
                def chooseModel(trial):
                    n_outputs = 1
                    NOLayers = trial.suggest_int('NOlayers', 1, 3)
                    model = Sequential()
                    for i in range(NOLayers):
                        NOUnits=int(trial.suggest_int('NOUnits{}'.format(i), 16, 1024))
                        rateDropout = trial.suggest_uniform('rateDropout{}'.format(i), 0.0, 0.5)
                        activation = trial.suggest_categorical('activation{}'.format(i), ['hard_sigmoid', 'linear', 'relu', 'sigmoid', 'softplus','softsign', 'tanh'])
                        model.add(Dense(NOUnits, activation=activation))
                        model.add(Dropout(rateDropout))
                    model.add(Dense(n_outputs,activation='sigmoid'))
                    return model
                def chooseOptimizer(trial):
                    nameOptimizer = trial.suggest_categorical('optimizer', ['adam'])
                    if nameOptimizer == 'adam':
                        lrAdam = trial.suggest_loguniform('lrAdam', 1e-5, 1e-2)
                        beta_1Adam = trial.suggest_uniform('beta1Adam', 0.9, 1)
                        beta_2Adam = trial.suggest_uniform('beta2Adam', 0.999, 1)
                        epsilonAdam = trial.suggest_loguniform('epsilonAdam', 1e-10, 1e-5)
                        opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
                    return opt
                verbose, epochs, sizeBatch = 0, 10000, trial.suggest_int("sizeBatch", 64, 1024)
                n_features, n_outputs = xTrain.shape[1], 1
                model = chooseModel(trial)
                opt = chooseOptimizer(trial)
                model.build((None,n_features))
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
                cbEarlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
                #cbReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=trial.suggest_uniform('lrDecay', 0.1, 0.5), patience=trial.suggest_int('lrDecayPatiance', 2, 50), min_lr=1e-5)
                #history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xValid, yValid), callbacks=[cbEarlyStopping, cbReduceLR])
                history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xValid, yValid), callbacks=[cbEarlyStopping])
                self.saveGraphTrain(history, trial.number)
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
                #全体のログをtxt形式で出力
                with open(os.path.join(self.dirResults, "results.txt"), mode='a') as f:
                    f.write(str(score)+","+str(trial.params)+'\n')
                return score
            if modelAlgorithm=="RF":
                return objectiveRF
            elif modelAlgorithm=="DNN":
                return objectiveDNN
            else:
                raise Exception("modelAlgorithm must be RF or DNN")
        study = optuna.create_study()
        study.optimize(set_objective(xTrain, yTrain, xValid, yValid, modelAlgorithm), timeout = 60*60*10)

    def test(self, xTrain, yTrain, xTest, yTest, modelAlgorithm, hp):
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
                    self.lossBest=loss
                    ysPredictedLabel=[]
                    for yPredicted in ysPredicted:
                        if yPredicted<0.5:
                            ysPredictedLabel.append(0)
                        else:
                            ysPredictedLabel.append(1)
                    #print(ysPredictedLabel)
                    analyzeResult(ysPredictedLabel, ys)
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
        def testDNN():
            n_outputs = 1
            model = Sequential()
            for i in range(int(hp["NOlayers"])):
                model.add(Dense(hp["NOUnits"+str(i)], activation=hp["activation"+str(i)]))
                model.add(Dropout(hp["rateDropout"+str(i)]))
                model.add(Dense(n_outputs,activation='sigmoid'))

            resultsValid={"loss":[],"mae":[], "acc":[],"detail":[[]]}

            verbose, epochs, sizeBatch = 1, 10000, hp["sizeBatch"]
            n_features, n_outputs = xTrain.shape[1], 1

            model.build((None,n_features))
            if hp["optimizer"] == "sgd":
                lrSgd = hp["lrSgd"]
                momentumSgd = hp["momentumSgd"]
                opt = keras.optimizers.SGD(lr=lrSgd, momentum=momentumSgd)
            elif hp["optimizer"] == "adagrad":
                opt = keras.optimizers.Adagrad()
            elif hp["optimizer"] == "adadelta":
                opt = keras.optimizers.Adadelta()
            elif hp["optimizer"] == 'adam':
                lrAdam = hp["lrAdam"]
                beta_1Adam = hp["beta1Adam"]
                beta_2Adam = hp["beta2Adam"]
                epsilonAdam = hp["epsilonAdam"]
                opt = keras.optimizers.Adam(lr=lrAdam, beta_1=beta_1Adam, beta_2=beta_2Adam, epsilon=epsilonAdam)
            elif hp["optimizer"] == "adamax":
                lrAdamax = hp["lrAdamax"]
                beta_1Adamax = hp["beta1Adamax"]
                beta_2Adamax = hp["beta2Adamax"]
                epsilonAdamax = hp["epsilonAdamax"]
                opt = keras.optimizers.Adamax(lr=lrAdamax, beta_1=beta_1Adamax, beta_2=beta_2Adamax, epsilon=epsilonAdamax)
            elif hp["optimizer"] == "nadam":
                lrNadam = hp["lrNadam"]
                beta_1Nadam = hp["beta1Nadam"]
                beta_2Nadam = hp["beta2Nadam"]
                epsilonNadam = hp["epsilonNadam"]
                opt = keras.optimizers.Nadam(lr=lrNadam, beta_1=beta_1Nadam, beta_2=beta_2Nadam, epsilon=epsilonNadam)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

            history=model.fit(xTrain, yTrain, epochs=epochs, batch_size=sizeBatch, verbose=verbose, validation_data=(xTest, yTest), callbacks=[TestCallback(xTest, yTest, sizeBatch, resultsValid, 0)])
            self.saveGraphTrain(history, 0)
        def testRF():
            for i in range(100):
                model=RandomForestClassifier(
                    n_estimators=hp["n_estimators"],
                    max_depth=hp["max_depth"],
                    max_leaf_nodes=hp["max_leaf_nodes"],
                    min_samples_leaf=hp["min_samples_leaf"],
                    min_samples_split=hp["min_samples_split"],
                    random_state=i)
                model.fit(xTrain,yTrain)
                ysPredicted=model.predict(xTest)
                analyzeResult(ysPredicted, yTest)
        if modelAlgorithm=="RF":
            testRF()
        elif modelAlgorithm=="DNN":
            testDNN()
        else:
            raise Exception("modelAlgorithm must be RF or DNN")