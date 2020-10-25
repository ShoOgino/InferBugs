import csv 

# isBuggyの1を読み込む
isBuggy=[]
with open("isBuggy.csv", encoding="utf_8") as f:
    reader = csv.reader(f)
    isBuggy.extend([row[0] for row in reader if(row[1]=="1")])
    print(len(isBuggy))

# hasBeenBuggyの1を読み込む
hasBeenBuggy=[]
with open("hasBeenBuggy.csv", encoding="utf_8") as f:
    reader = csv.reader(f)
    hasBeenBuggy.extend([row[0] for row in reader if(row[1]=="1")])
    print(len(hasBeenBuggy))

# isBuggyにとってのbuggyが、hasBeenBuggyでbuggyかを調べる。そうであれば

count=0
for row in isBuggy:
    if(row in hasBeenBuggy):
        print(row)
        count=count+1
print(count/len(isBuggy))