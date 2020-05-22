import os
import sys
import glob
import csv
from pydriller import GitRepository, RepositoryMining, domain
import pandas as pd
import json
import datetime
import subprocess

from src.dataset import Dataset
from src.utility import Option, UtilPath

# optionの構造を理解しているのはここだけ。
class Maneger:
    def __init__(self, optionInputted):
        print(json.dumps(optionInputted,indent=4))
        self.option = Option(optionInputted)

    def do(self):
        if(self.option.mode == "train"):
            dataset=Dataset(self.option.getRepositorieImproved())
            dataset.prepare()
            dataset.save(UtilPath.Dataset(self.option.name))
            #model=Model()
            #model.prepare(){features, architecture, hyper-parameters}
            #model.save()
        elif(self.option.mode == "infer"):
            #dataset=Dataset()
            #dataset.prepare()
            #dataset.save()
            #model=Model()
            #model.prepare()
            #model.infer()g
            pass