import sys
import os
import csv
import glob

import pandas as pd
import numpy as np
import re
def check():
    pathsCorrect=[]
    with open('method.csv', 'r',newline="") as f:
        reader=csv.reader(f)
        rows=[row for row in reader]
        for i,row in enumerate(rows):
            if i==0:
                continue
            classes=[]
            patternClass=r'[^/]+(?=\.c)'
            patternNamePre=r'[^/]+(?=\.f)'
            patternNamePost=r'[^/]+$'
            classesFound=re.findall(patternClass,row[18])
            if (2<=len(classesFound)):
                for i in range(1, len(classesFound)):
                    classes.append(classesFound[i])
            namePre=re.search(patternNamePre, row[18])
            if namePre:
                directory=row[18][:namePre.start()]
                namePre=namePre.group()

            namePost=re.search(patternNamePost, row[18])
            if namePost:
                namePost=namePost.group()
            pathCorrect=os.path.join(directory, namePre)
            for _class in classes:
                pathCorrect=pathCorrect+"."
                pathCorrect=pathCorrect+_class
            pathCorrect=pathCorrect+"#"+namePost.replace("@",",")+".mjava"
            pathsCorrect.append(pathCorrect)
    count=0
    pathsNo=[]
    for pathCorrect in pathsCorrect:
        path=os.path.join("C:/Users/login/work/InferBugs/actions/test/testDataset/ant/repository",pathCorrect).replace("\\","/")
        if not os.path.exists(path):
            pathsNo.append([path])
            count=count+1
    with open('pathsNo.csv', 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(pathsNo)
    print(count)

def rewrite():
    rowsCorrect=[]
    with open('period.csv', 'r',newline="") as f:
        reader=csv.reader(f)
        rows=[row for row in reader]
        for row in rows:
            print()
            path=row[1]
            patternNamePre=r'[^/]+(?=\.)'
            patternNamePost=r'[^/]+$'
            directory=""

            namePre=re.search(patternNamePre, path)
            if namePre:
                directory=path[:namePre.start()]
                namePre=namePre.group()
                print(namePre)

            path=row[1]
            namePost=re.search(patternNamePost, path)
            if namePost:
                namePost=namePost.group()
                print(namePost)
            pathCorrect=os.path.join(directory,namePre+"#"+namePost.replace("@",",")+".mjava")
            rowsCorrect.append([row[0],pathCorrect])
    with open('_period.csv', 'w',newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rowsCorrect)
check()