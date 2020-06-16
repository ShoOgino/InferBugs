import sys
import os
import csv
import glob

import pandas as pd
import numpy as np
import re

def makeTest():
    pathDataCorrect=sys.argv[1]
    df = pd.read_csv(pathDataCorrect)
    for i, column in enumerate(df.columns):
        if column=="path":
            continue
        print(column)
        valuesUnique=np.sort(df[column].unique())
        valueMax=valuesUnique[len(valuesUnique)-1]
        valueMid=valuesUnique[1]
        valueMin=valuesUnique[0]
        with open(os.path.join(os.path.dirname(pathDataCorrect), column+'.csv'), 'w',newline="") as f:
            writer = csv.writer(f)
            #valueMin = df.at[df[df[column]==valueMin].head(1).index[0],column]
            #valueMid = df.at[df[df[column]==valueMid].head(1).index[0],column]
            #valueMax = df.at[df[df[column]==valueMax].head(1).index[0],column]

            #patternForeign=r'/\d/'
            #patternAt=r'@'
            #pathValueMin = re.sub(patternForeign, "/", df.at[df[df[column]==valueMin].head(1).index[0],"path"])
            #pathValueMin = re.sub(patternAt, ",", pathValueMin)
            #pathValueMid = re.sub(patternForeign, "/", df.at[df[df[column]==valueMid].head(1).index[0],"path"])
            #pathValueMid = re.sub(patternAt, ",", pathValueMid)
            #pathValueMax = re.sub(patternForeign, "/", df.at[df[df[column]==valueMax].head(1).index[0],"path"])
            #pathValueMax = re.sub(patternAt, ",", pathValueMax)

            rowMin=df[df[column]==valueMin].head(1).values[0].tolist()
            rowMin[18]=rename(rowMin[18])
            rowMid=df[df[column]==valueMid].head(1).values[0].tolist()
            rowMid[18]=rename(rowMid[18])
            rowMax=df[df[column]==valueMax].head(1).values[0].tolist()
            rowMax[18]=rename(rowMax[18])
            writer.writerow(df[df[column]==valueMin].head(1))
            writer.writerow(rowMin)
            writer.writerow(rowMid)
            writer.writerow(rowMax)

           #writer.writerow([valueMin ,pathValueMin])
           #writer.writerow([valueMid, pathValueMid])
           #writer.writerow([valueMax, pathValueMax])
           #for k in range(5):
           #    index=list(df.sample().index)[0]
           #    pathRandom = re.sub(patternForeign, "/", df.iat[index,18])
           #    pathRandom = re.sub(patternAt, ",", pathRandom)
           #    writer.writerow([df.iat[index,i],pathRandom])

    #rows_sorted = sorted(rows, key=lambda x: x[0])
    #print(rows_sorted)
def rename(str):
    classes=[]
    patternClass=r'[^/]+(?=\.c)'
    patternPreName=r'[^/]+(?=\.f)'
    patternName=r'[^/]+$'
    classesFound=re.findall(patternClass,str)

    if (2<=len(classesFound)):
        for i in range(1, len(classesFound)):
            classes.append(classesFound[i])
    preName=re.search(patternPreName, str)
    if preName:
        directory=str[:preName.start()]
        namePre=preName.group()

    name=re.search(patternName, str)
    if name:
        name=name.group()
    pathCorrect=os.path.join(directory, namePre)
    for _class in classes:
        pathCorrect=pathCorrect+"."
        pathCorrect=pathCorrect+_class
    pathCorrect=pathCorrect+"#"+name.replace("@",",")+".mjava"
    return pathCorrect

makeTest()