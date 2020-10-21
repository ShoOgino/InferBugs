from src.manager import Maneger
import datetime

option={
    "date"              : datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
    "project"           : "linuxtools",
    "release4test"      : "4",
    "variableDependent" : "isBuggy",
    "purpose"           : "search",
    "modelAlgorithm"    : "DNN",
    "pathModel"         : "",
    "pathHP"            : "",
}

option["idExperiment"] = \
        option["project"] + "_" \
        + option["variableDependent"] + "_" \
        + option["release4test"] + "_" \
        + option["modelAlgorithm"] + "_" \
        + option["date"]

maneger = Maneger(option)
maneger.do()