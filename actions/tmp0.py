import sys
import os
import csv
import glob

import pandas as pd
import numpy as np
import re
def check():
    with open('method.csv', 'r',newline="") as f:
        reader=csv.reader(f)
        rows=[row for row in reader]
        for i,row in enumerate(rows):
            pattern=r'[^/]+(?=\.c)'
            list=re.findall(pattern,row[18])
            print(row[18])
            if (2<=len(list)):
                print(i)

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