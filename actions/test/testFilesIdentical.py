def checkThereAreIdenticalFiles():
    list=[]
    with open(sys.argv[2]) as f:
        reader = csv.reader(f)
        for row in reader:
            if os.path.exists("./"+row[18]):
                pass
            else:
                pass
                #print(row[18])

    with open(sys.argv[2]) as f:
        reader = csv.reader(f)
        for row in reader:
            if os.path.exists("./"+row[18]):
                list.append(os.path.abspath("./"+row[18]))

    strSearch=sys.argv[1]+"/**/*.java"
    listFile=glob.glob(strSearch, recursive=True)
    for filename in listFile:
        if (not os.path.abspath(filename) in list):
            pass
            print(filename)
    df =pd.read_csv(sys.argv[2])
    print("files in csv: "+ str(len(df)))
    print("files in working tree: "+str(len(listFile)))
