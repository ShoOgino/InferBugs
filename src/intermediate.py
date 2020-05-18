import os
import sys
import glob
import csv
from pydriller import GitRepository, RepositoryMining, domain
import pandas as pd
import json
import datetime
from tqdm import tqdm
import dataset

def intermediate(pathRepository, nameExperiment):
    gr = GitRepository(pathRepository)
    pathsFile=glob.glob(pathRepository+"/**/*.java", recursive=True)
    json_open=open("../dataset/"+nameExperiment+"/annotationsReformatted.json", 'r')
    formattedAnnotations=json.load(json_open)

    with open('../result/'+nameExperiment+"/"+"["+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+"]"+'result.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["LOC", "addLOC", "delLOC", "chgNum", "fixChgNum", "pastBugNum", "hcm", "devTotal", "devMinor", "devMajor",     "Ownership", "period", "avgInterval", "maxInterval", "minInterval", "logCoupNum", "bugIntroNum", "isBuggy", "file"])

        with tqdm(pathsFile,bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")     as  pbar:
            for pathFile in pbar:
                pathFile=os.path.abspath(pathFile)
                nameFile=os.path.basename(pathFile)
                pbar.postfix=nameFile

                file=dataset.File()

                commits = gr.get_commits_modified_file(pathFile)
                writer.writerow([
                    file.calculateLOC(pathFile),
                    file.calculateAddLOC(gr,commits,nameFile),
                    file.calculateDelLOC(gr,commits,nameFile),
                    file.calculateChgNum(gr,commits,nameFile),
                    file.calculateFixChgNum(gr, commits, nameFile),
                    file.calculatePastBugNum(gr, commits, nameFile, formattedAnnotations),
                    file.calculateHCM(gr, commits, nameFile),
                    file.calculateDevTotal(gr,commits,nameFile),
                    file.calculateDevMinor(gr,commits,nameFile),
                    file.calculateDevMajor(gr,commits,nameFile),
                    file.calculateOwnership(gr,commits,nameFile),
                    file.calculatePeriod(gr,commits,nameFile),
                    file.calculateAvgInterval(gr,commits,nameFile),
                    file.calculateMaxInterval(gr,commits,nameFile),
                    file.calculateMinInterval(gr,commits,nameFile),
                    file.calculateLogCoupNum(gr, commits, nameFile, formattedAnnotations),
                    file.calculateBugIntroNum(gr,commits,nameFile,formattedAnnotations),
                    file.calculateIsBuggy(gr,commits,nameFile, formattedAnnotations),
                    pathFile
                    ]
                )