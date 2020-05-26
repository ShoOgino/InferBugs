import sys
import os
import csv
import glob

import pandas as pd
import numpy as np

def makeTest():
    pathDataCorrect=sys.argv[1]
    df = pd.read_csv(pathDataCorrect)
    for column in df.columns:
        if column=="path":
            continue
        print(column)
        valuesUnique=np.sort(df[column].unique())
        valueMax=valuesUnique[len(valuesUnique)-1]
        valueMid=valuesUnique[1]
        valueMin=valuesUnique[0]
        with open(column+'.csv', 'w',newline="") as f:
           writer = csv.writer(f)
           writer.writerow([df.at[df[df[column]==valueMin].head(1).index[0],column],df.at[df[df[column]==valueMin].head(1).index[0],"path"]])
           writer.writerow([df.at[df[df[column]==valueMid].head(1).index[0],column],df.at[df[df[column]==valueMid].head(1).index[0],"path"]])
           writer.writerow([df.at[df[df[column]==valueMax].head(1).index[0],column],df.at[df[df[column]==valueMax].head(1).index[0],"path"]])
           if column=="period":
               for i in range(30):
                    index=list(df.sample().index)[0]
                    writer.writerow([df.iat[index,11],df.iat[index,18]])
        
    #rows_sorted = sorted(rows, key=lambda x: x[0])
    #print(rows_sorted)
makeTest()