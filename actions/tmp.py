import csv
from  src.utility import UtilPath
import os

with open(os.path.join(UtilPath.Test(), 'cassandra_method.csv')) as f:
    reader = csv.reader(f)
    for row in reader:
        print(row[18])
