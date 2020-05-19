import os
import sys
import glob
import csv
from pydriller import GitRepository, RepositoryMining, domain
import pandas as pd
import json
import datetime
from src.dataset import Dataset, UtilityMetrics
import subprocess

def intermediate(nameExperiment, identifiersDataset, parameterTrain):
    for identifierDataset in identifiersDataset:
        pathData=os.path.join("../dataset",nameExperiment,identifierDataset["nameRepository"])
        dataset=Dataset(identifierDataset["urlRepository"], pathData, identifierDataset["nameRepository"], identifierDataset["filterFile"], identifierDataset["codeIssueJira"], identifierDataset["projectJira"])
        dataset.calculateDataset()
        print(dataset.data)
        #   with open('../result/'+nameExperiment+"/"+"["+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+"]"+'result.csv', 'a', newline="") as f:
        #       writer = csv.writer(f)
        #       writer.writerow(["LOC", "addLOC", "delLOC", "chgNum", "fixChgNum", "pastBugNum", "hcm", "devTotal", "devMinor", "devMajor", "Ownership", "period", "avgInterval", "maxInterval", "minInterval", "logCoupNum", "bugIntroNum", "isBuggy", "file"])


