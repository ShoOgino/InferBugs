import os
import re
import ast
import json

path = "./"
files = os.listdir(path)
dirs = [f for f in files if os.path.isdir(os.path.join(path, f))]
for dir in dirs:
    print(dir)
    lossBest=10000000000
    hpBest={}
    pathResults=os.path.join(dir, "results.txt")
    modelAlgorithm=""
    if "RF" in dir:
        modelAlgorithm="RF"
    elif ("DNN" in dir) or ("DL" in dir):
        modelAlgorithm="DNN"
    else:
        raise Exception("modelAlgorithm must be RF or DNN")
    if os.path.exists(pathResults):
        with open(pathResults, encoding='utf-8') as f:
            lines = f.readlines()
            pattern = ','
        for line in lines:
            value, hp= re.split(pattern, line, 1)
            if float(value)<lossBest:
                print(value + "is lower than" + str(lossBest))
                lossBest = float(value)
                hpBest=ast.literal_eval(hp)
                print(hpBest)
                with open(os.path.join(dir,"hp"+modelAlgorithm+".json"), 'w') as f:
                    json.dump(hpBest, f, indent=4)