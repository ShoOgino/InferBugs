import sys
import os
import csv
import glob

import pandas as pd
import numpy as np
import re
def check():
    pathsCorrect=[]
    pathDataTest=sys.argv[1]
    with open(pathDataTest, 'r',newline="") as f:
        reader=csv.reader(f)
        rows=[row for row in reader]
        for i,row in enumerate(rows):
            if i==0:
                continue
            classes=[]
            patternClass=r'[^/]+(?=\.c)'
            patternPreName=r'[^/]+(?=\.f)'
            patternName=r'[^/]+$'
            classesFound=re.findall(patternClass,row[18])

            if (2<=len(classesFound)):
                for i in range(1, len(classesFound)):
                    classes.append(classesFound[i])
            preName=re.search(patternPreName, row[18])
            if preName:
                directory=row[18][:preName.start()]
                namePre=preName.group()

            name=re.search(patternName, row[18])
            if name:
                name=name.group()
            pathCorrect=os.path.join(directory, namePre)
            for _class in classes:
                pathCorrect=pathCorrect+"."
                pathCorrect=pathCorrect+_class
            pathCorrect=pathCorrect+"#"+name.replace("@",",")+".mjava"
            pathsCorrect.append(pathCorrect)
            print(pathCorrect)
    #with open('loc.csv', 'w',newline="") as f:
    #    writer = csv.writer(f)
    #    writer.writerows()

check()