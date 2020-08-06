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


dirModel="../../models"
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


def getModelBest():
    #features2Drop, parametersBest=getParametersHyperBest()
    features2Drop=['NOCharacters', 'NOWordsNum', 'NOWordsConjunction', 'NOWordsUnregistered', 'NOCharactersKanji']
    parametersBest={
        "n_estimators":2,
        "max_depth":3,
        "min_samples_leaf":3,
        "min_samples_split":2,
        "random_state":7
    }
    #getParametersTrainBest(getDatasetsSplit(features2Drop))
    testParametersHyper(features2Drop, parametersBest)
    dataset=getDatasetBest()
    model=RandomForestClassifier(
        n_estimators=parametersBest["n_estimators"],
        max_depth=parametersBest["max_depth"],
        min_samples_leaf=parametersBest["min_samples_leaf"],
        min_samples_split=parametersBest["min_samples_split"],
        random_state=parametersBest["random_state"]
    )
    model.fit(dataset[0]["train_x"], dataset[0]["train_y"])
    return model

def getDatasetBest():
    features2Drop=['NOCharacters', 'NOWordsNum', 'NOWordsConjunction', 'NOWordsUnregistered', 'NOCharactersKanji']
    return getDatasetsSplit(features2Drop)

def testModel(model, datasetsSplit):
    print("testing model...")
    print(datasetsSplit[0]["test_y"][0])
    print(model.predict(datasetsSplit[0]["test_x"]))
    accuracyTest=accuracy_score(datasetsSplit[0]["test_y"], model.predict(datasetsSplit[0]["test_x"]))
    print("finish testing model")
    print("accuracy: "+str(accuracyTest))

def testParametersHyper(features2Drop, parametersBest):
    datasetsSplit=getDatasetsSplit(features2Drop)
    accuraciesTrain=[]
    accuraciesValid=[]
    for patternSplit in range(10):
        train_x=datasetsSplit[patternSplit]["train_x"]
        train_y=datasetsSplit[patternSplit]["train_y"]
        valid_x=datasetsSplit[patternSplit]["valid_x"]
        valid_y=datasetsSplit[patternSplit]["valid_y"]
        model=RandomForestClassifier(
            n_estimators=parametersBest["n_estimators"],
            max_depth=parametersBest["max_depth"],
            min_samples_leaf=parametersBest["min_samples_leaf"],
            min_samples_split=parametersBest["min_samples_split"],
            random_state=parametersBest["random_state"]
        )
        model.fit(train_x, train_y)
        accuraciesTrain.append(model.score(train_x, train_y))
        accuraciesValid.append(accuracy_score(valid_y, model.predict(valid_x)))

    accuracyTrainAverage=sum(accuraciesTrain)/len(accuraciesTrain)
    accuracyValidAverage=sum(accuraciesValid)/len(accuraciesValid)
    print("averageTrain:"+str(accuracyTrainAverage))
    print("averageValid:"+str(accuracyValidAverage))


def getDatasetsSplit(*features2Drop):
    dataset=[]
    datasetSplit=[]
    for i in range(10):
        with open('dataset'+str(i)+'.json') as f:
            d = json.load(f)
        dataset.append(d)

    for trial in range(10):
        dictionary={}
        train={
            "NOSentences":[],
            "NOWords" : [],
            "NOWordsEnglish" : [],
            "NOWordsNum" : [],
            "NOWordsParenthese" : [],
            "NOWordsReadingPoint" : [],
            "NOWordsConjunction" : [],
            "NOWordsPostpositionalParticleNo" : [],
            "NOWordsUnregistered" : [],
            "NOCharacters" : [],
            "NOCharactersKanji" : [],
            "isReadable" : []
        }
        valid={
            "NOSentences":[],
            "NOWords" : [],
            "NOWordsEnglish" : [],
            "NOWordsNum" : [],
            "NOWordsParenthese" : [],
            "NOWordsReadingPoint" : [],
            "NOWordsConjunction" : [],
            "NOWordsPostpositionalParticleNo" : [],
            "NOWordsUnregistered" : [],
            "NOCharacters" : [],
            "NOCharactersKanji" : [],
            "isReadable" : []
        }
        for i in range(10):
            for row in dataset[i]:
                for column in dataset[i][row]:
                    if(i==trial):
                        valid[column].append(dataset[i][row][column])
                    else:
                        train[column].append(dataset[i][row][column])
        test={
            "NOSentences":[],
            "NOWords" : [],
            "NOWordsEnglish" : [],
            "NOWordsNum" : [],
            "NOWordsParenthese" : [],
            "NOWordsReadingPoint" : [],
            "NOWordsConjunction" : [],
            "NOWordsPostpositionalParticleNo" : [],
            "NOWordsUnregistered" : [],
            "NOCharacters" : [],
            "NOCharactersKanji" : [],
            "isReadable" : []
        }
        with open('dataset'+str(10)+'.json') as f:
            d = json.load(f)
            for row in d:
                for column in d[row]:
                    test[column].append(d[row][column])
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        features=list(features2Drop)
        train_x = pd.DataFrame(train)
        train_x = train_x.drop(['isReadable'], axis=1)
        for feature in features:
            train_x=train_x.drop(feature, axis=1)
        #train_x = (train_x - train_x.mean())/ train_x.std(ddof=0) #RFでは正規化しない。
        train_y = pd.DataFrame(train)['isReadable']

        valid_x = pd.DataFrame(valid)
        valid_x = valid_x.drop(['isReadable'], axis=1)
        for feature in features:
            valid_x=valid_x.drop(feature, axis=1)
        valid_y = pd.DataFrame(valid)['isReadable']

        test_x = pd.DataFrame(test)
        test_x = test_x.drop(['isReadable'], axis=1)
        for feature in features:
            test_x=test_x.drop(feature, axis=1)
        test_y = pd.DataFrame(test)['isReadable']
        dictionary["train_x"]=train_x
        dictionary["train_y"]=train_y
        dictionary["valid_x"]=valid_x
        dictionary["valid_y"]=valid_y
        dictionary["test_x"]=test_x
        dictionary["test_y"]=test_y
        datasetSplit.append(dictionary)
    return datasetSplit


def getParametersHyperBest():
    features=[
        "NOSentences",
        "NOWords",
        "NOWordsEnglish",
        "NOWordsNum",
        "NOWordsParenthese",
        "NOWordsReadingPoint",
        "NOWordsConjunction",
        "NOWordsPostpositionalParticleNo",
        "NOWordsUnregistered",
        "NOCharacters",
        "NOCharactersKanji",
    ]
    accuracies=[]
    features2Drop=[]
    for trialFeature in range(10):
        accuracyBest=0
        featureWorst=""
        for index, feature in enumerate(features):
            print(features2Drop)
            print(feature)
            datasetsSplit = getDatasetsSplit(features2Drop+[feature])
            parametersBest, accuracy=getParametersTrainBest(datasetsSplit)
            if(accuracyBest<accuracy):
                accuracyBest = accuracy
                featureWorst = feature
        accuracies.append(accuracyBest)
        features2Drop.append(featureWorst)
        features.remove(featureWorst)
    print(features2Drop)
    print(accuracies)
    max_idx=accuracies.index(max(accuracies))
    datasetsSplit = getDatasetsSplit(features2Drop[:max_idx])
    parametersBest, accuracy = getParametersTrainBest(datasetsSplit)
    return features2Drop[:max_idx], parametersBest


def getParametersTrainBest(datasetsSplit):
    accuracyBest=0
    parametersBest={
        "n_estimators":0 ,
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
                        print("n_estimators: "+str(n_estimators))
                        print("max_depth: "+str(max_depth))
                        print("min_samples_leaf: "+str(min_samples_leaf))
                        print("min_samples_split: "+str(min_samples_split))
                        print("random_state: "+str(random_state))
                        accuraciesTrain=[]
                        accuraciesValid=[]
                        for patternSplit in range(10):
                            train_x=datasetsSplit[patternSplit]["train_x"]
                            train_y=datasetsSplit[patternSplit]["train_y"]
                            valid_x=datasetsSplit[patternSplit]["valid_x"]
                            valid_y=datasetsSplit[patternSplit]["valid_y"]
                            model=RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                random_state=random_state
                            )
                            model.fit(train_x, train_y)
                            accuraciesTrain.append(model.score(train_x, train_y))
                            accuraciesValid.append(accuracy_score(valid_y, model.predict(valid_x)))

                        accuracyTrainAverage=sum(accuraciesTrain)/len(accuraciesTrain)
                        accuracyValidAverage=sum(accuraciesValid)/len(accuraciesValid)
                        print("averageTrain:"+str(accuracyTrainAverage))
                        print("averageValid:"+str(accuracyValidAverage))
                        if(accuracyBest<accuracyValidAverage):
                            accuracyBest=accuracyValidAverage
                            parametersBest={
                                "n_estimators":n_estimators ,
                                "random_state":random_state,
                                "max_depth":max_depth,
                                "min_samples_leaf":min_samples_leaf,
                                "min_samples_split":min_samples_split
                            }
    print(accuracyBest)
    print(parametersBest)
    return parametersBest, accuracyBest

def main():
    model=getModelBest()
    dataset=getDatasetBest()
    testModel(model, dataset)

if __name__ == '__main__':
    main()
