import csv
import json 

d={}
with open('test/cas_file.csv') as f:
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if(i==0):
            continue
        d[row[18]]={}
        d[row[18]]["LOC"]=row[0]
        d[row[18]]["addLOC"]=row[1]
        d[row[18]]["delLOC"]=row[2]
        d[row[18]]["chgNum"]=row[3]
        d[row[18]]["fixChgNum"]=row[4]
        d[row[18]]["pastBugNum"]=row[5]
        d[row[18]]["period"]=row[11]
        d[row[18]]["bugIntroNum"]=row[16]
        d[row[18]]["logCoupNum"]=row[15]
        d[row[18]]["avgInterval"]=row[12]
        d[row[18]]["maxInterval"]=row[13]
        d[row[18]]["minInterval"]=row[14]
        d[row[18]]["HCM"]=row[6]
        d[row[18]]["isBuggy"]=row[17]
        d[row[18]]["devTotal"]=row[7]
        d[row[18]]["devMinor"]=row[8]
        d[row[18]]["devMajor"]=row[9]
        d[row[18]]["ownership"]=row[10]
        if(i==10):
            break
with open('test.json', 'w') as f:
    json.dump(d, f, indent=4)
