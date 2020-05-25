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
            if os.path.exists(pathFile):
                listPathsCorrect.append(os.path.abspath(pathFile))

    strSearch = os.path.abspath(pathFolderRepository) + "/**/*.java"
    listPaths = glob.glob(strSearch, recursive=True)

    print(listPathsCorrect)
    #print(listPaths)

    #for pathCorrect in listPathsCorrect:
    #    if (not pathCorrect in listPaths):
    #        print(pathCorrect)
    for path in listPaths:
        if (not path in listPathsCorrect):
            print(path)


    print("files in csv: "+ str(len(listPathsCorrect)))
    print("files in working tree: "+str(len(listPaths)))

checkRepositoryHasCorrectFiles()