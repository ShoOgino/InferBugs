import os
import sys
import glob
import pandas as pd
import json
import datetime
import subprocess
import tensorflow as tf

from src.modeler import Modeler
from src.utility import UtilPath
from src.dataset import Dataset

# optionの構造を理解しているのはここだけ。
class Maneger:
    def __init__(self, option):
        print(json.dumps(option,indent=4))
        self.option = option

    def checkGPU(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")

    def do(self):
        self.checkGPU()
        os.makedirs(UtilPath.ResultAction(self.option["idExperiment"]), exist_ok=True)
        modeler=Modeler(UtilPath.ResultAction(self.option["idExperiment"]))
        dataset=Dataset(self.option["project"], self.option["variableDependent"], self.option["release4test"], self.option["purpose"])
        if self.option["purpose"]=="test":
            xTrain4Test, yTrain4Test = dataset.getTrain4Test()
            xTest4Test, yTest4Test = dataset.getTest4Test()
            json_open = open(self.option["pathHP"], 'r')
            hp = json.load(json_open)
            modeler.test(xTrain4Test, yTrain4Test, xTest4Test, yTest4Test, self.option["modelAlgorithm"], hp)
        elif self.option["purpose"]=="search":
            xTrain4Search, yTrain4Search = dataset.getTrain4Search()
            xValid4Search, yValid4Search = dataset.getValid4Search()
            modeler.search(xTrain4Search, yTrain4Search, xValid4Search, yValid4Search, self.option["modelAlgorithm"], self.option["time2search"])