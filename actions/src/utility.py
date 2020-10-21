from urllib.parse import quote
import json
import urllib.request as url
import sys
import os
import re
import statistics
import math
import glob
import pathlib
import datetime

class UtilPath:
    @staticmethod
    def Root():
        return pathlib.Path(__file__).parent.parent.parent.resolve()


    @staticmethod
    def Docs():
        return os.path.join(UtilPath.Root(),"docs")


    @staticmethod
    def Actions():
        return os.path.join(UtilPath.Root(),"actions")
    @staticmethod
    def Src():
        return os.path.join(UtilPath.Actions(),"src")
    @staticmethod
    def Lib():
        return os.path.join(UtilPath.Src(),"lib")
    @staticmethod
    def Test():
        return os.path.join(UtilPath.Actions(),"test")


    @staticmethod
    def Datasets():
        return os.path.join(UtilPath.Root(),"datasets")
    @staticmethod
    def DatasetAction(nameExperiment):
        return os.path.join(UtilPath.Datasets(),nameExperiment)
    @staticmethod
    def AboutRepository(nameExperiment, nameRepository):
        return os.path.join(UtilPath.DatasetAction(nameExperiment),nameRepository)
    @staticmethod
    def Repository(nameExperiment, nameRepository):
        return os.path.join(UtilPath.AboutRepository(nameExperiment,nameRepository),"repository")
    @staticmethod
    def Dataset(nameExperiment):
        return os.path.join(UtilPath.DatasetAction(nameExperiment),"dataset.json")


    @staticmethod
    def Models():
        return os.path.join(UtilPath.Root(),"models")
    @staticmethod
    def ModelAction(nameExperiment):
        return os.path.join(UtilPath.Models(),nameExperiment)
    @staticmethod
    def Model(nameExperiment):
        return os.path.join(UtilPath.ModelAction(nameExperiment),str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+".model")


    @staticmethod
    def Results():
        return os.path.join(UtilPath.Root(),"results")
    @staticmethod
    def ResultAction(nameExperiment):
        return os.path.join(UtilPath.Results(),nameExperiment)
    @staticmethod
    def LogOverall(nameExperiment, mode):
        return os.path.join(UtilPath.ResultAction(nameExperiment),"["+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+"]"+mode+"_overall.log")
    @staticmethod
    def Result(nameExperiment, mode):
        return os.path.join(UtilPath.ResultAction(nameExperiment),"["+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+"]"+mode+"_result.csv")
