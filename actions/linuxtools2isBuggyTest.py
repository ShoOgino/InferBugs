from src.manager import Maneger
from src.utility import UtilPath
import datetime

option={
    "date"              : datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
    "project"           : "linuxtools",
    "release4test"      : "2",
    "variableDependent" : "isBuggy",
    "purpose"           : "test",
    "modelAlgorithm"    : "DNN",
    "pathModel"         : "",
    "pathHP"            : UtilPath.Datasets()+"/"+"linuxtools/isBuggy/2/hpDNN.json"
}

option["idExperiment"] = \
        option["project"] + "_" \
        + option["variableDependent"] + "_" \
        + option["release4test"] + "_" \
        + option["modelAlgorithm"] + "_" \
        + option["date"]

maneger = Maneger(option)
maneger.do()