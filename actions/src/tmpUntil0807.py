import csv

with open('../../datasets/cassandra/1.0.csv', encoding = "utf_8") as f1:
    dataset = csv.reader(f1)
    for row in dataset:
        print(row)
