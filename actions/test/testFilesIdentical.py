import sys
import os
import csv
import glob
import pandas as pd

def checkRepositoryHasCorrectFiles():
    pathFolderRepository=sys.argv[1]
    pathFilePathsCorrect=sys.argv[2]
    listPathsCorrect=[]
    with open(pathFilePathsCorrect) as f:
        reader = csv.reader(f)
        for row in reader:
            pathFile=os.path.join(pathFolderRepository, row[18])
            #if os.path.exists(pathFile):
            listPathsCorrect.append(os.path.abspath(pathFile))

    strSearch = os.path.abspath(pathFolderRepository) + "/**/*.java"
    listPaths = glob.glob(strSearch, recursive=True)

    with open("log.txt",mode="w",newline="") as f:
        f.write("not in correct\n")
    count1=0
    for path in listPaths:
        if (not path in listPathsCorrect):
            count1=count1+1
            with open("log.txt",mode="a") as f:
                f.write(path)
                f.write("\n")
    
    count2=0
    with open("log.txt",mode="a",newline="") as f:
        f.write("not in real\n")
    for  path in listPathsCorrect:
        if (not path in listPaths):
            count2+=1
            with open("log.txt",mode="a") as f:
                f.write(path)
                f.write("\n")

    print(len(listPathsCorrect))
    print(len(listPaths))
    print(count1)
    print(count2)
checkRepositoryHasCorrectFiles()