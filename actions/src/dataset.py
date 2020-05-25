import glob
import os
import sys
from pydriller import GitRepository, RepositoryMining, domain
import csv
import json
import math
import statistics
from tqdm import tqdm

import datetime
import re

class Dataset:
    def __init__(self, repositories):
        self.dataset=[]
        self.repositories=repositories
    def prepare(self):
        for repository in self.repositories:
            datasetRepository=[]
            if(not os.path.exists(repository["path"])):
                pass
                #rawdata=Rawdata()
            gr = GitRepository(repository["path"])
            pathsFile = [pathFile for pathFile in glob.glob(repository["path"]+"/**/*.mjava", recursive=True) if re.match(repository["filterFile"], pathFile)]
            commitsBug=self.getCommitsBug(repository)

            with tqdm(pathsFile,bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]") as pbar:
                for pathFile in pbar:
                    nameFile=os.path.basename(pathFile)
                    pbar.postfix=nameFile
                    pbar.desc=repository["name"]
                    datasetRepository.append(Data(gr, pathFile, commitsBug).getData())
            self.dataset.extend(datasetRepository)
    def getCommitsBug(self, repository):
        commitsBug={}
        annotations=json.load(open(repository["pathAnnotations"], 'r'))
        for commit in annotations:
            for i in range(len(annotations[commit])):
                if(not commit in commitsBug):
                    commitsBug[commit]={"fix":[], "prefix":[], "intro":[]} # prefix を追加
                commitsBug[commit]["fix"].append(annotations[commit][i]["filePath"])
                for revision in annotations[commit][i]["revisions"]:
                    if(revision!=commit):
                        if(not revision in commitsBug):
                             commitsBug[revision]={"fix":[], "prefix":[], "intro":[]}
                        commitsBug[revision]["intro"].append(annotations[commit][i]["filePath"])
        return commitsBug
    def save(self, path):
        with open(path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.dataset)
    def load(self):
        pass
    def visualize(self):
        pass
class Data:
    def __init__(self, gr, pathFile, commitsBug):
        self.gr=gr
        self.pathFile=pathFile
        self.nameFile=os.path.basename(pathFile)
        self.commits =[]
        print(self.gr.get_commits_modified_file(self.pathFile))
        for commit in self.gr.get_commits_modified_file(self.pathFile):
            for modification in self.gr.get_commit(commit).modifications:
                if (self.nameFile == modification.filename):
                    self.commits.append(commit)
                    print(modification.change_type)
                    if (modification.change_type!=modification.change_type.MODIFY) and (modification.change_type!=modification.change_type.DELETE):
                        break
            else:
                continue
            break
        self.commitsBug=commitsBug

    def getData(self):
        return [
            self.calculateLOC(),
            self.calculateAddLOC(),
            self.calculateDelLOC(),
            self.calculateChgNum(),
            self.calculateFixChgNum(),
            self.calculatePastBugNum(),
            self.calculateHCM(),
            self.calculateDevTotal(),
            self.calculateDevMinor(),
            self.calculateDevMajor(),
            self.calculateOwnership(),
            self.calculatePeriod(),
            self.calculateAvgInterval(),
            self.calculateMaxInterval(),
            self.calculateMinInterval(),
            self.calculateLogCoupNum(),
            self.calculateBugIntroNum(),
            self.calculateIsBuggy(),
            self.pathFile
        ]
    def calculateLOC(self):
        with open(self.pathFile, "r") as fr:
            return len(fr.readlines())
    def calculateAddLOC(self):
        addLOC=0
        for commit in self.commits:
            for modification in self.gr.get_commit(commit).modifications:
                if (modification.filename==self.nameFile) and (modification.change_type==modification.change_type.MODIFY):
                    addLOC+=modification.added
        return addLOC
    def calculateDelLOC(self):
        delLOC=0
        for commit in self.commits:
            for modification in self.gr.get_commit(commit).modifications:
                if (modification.filename==self.nameFile) and (modification.change_type==modification.change_type.MODIFY):
                    delLOC+=modification.removed
        return delLOC
    def calculateChgNum(self):
        return len(self.commits)
    def calculateFixChgNum(self):
        pass
    def calculatePastBugNum(self):
        pastBugNum=0
        for commit in self.commits:
            if(commit in self.commitsBug):
                if(self.nameFile in self.commitsBug[commit]["fix"]):
                    pastBugNum=pastBugNum+1
        return pastBugNum
    def calculatePeriod(self):
        dates=[self.gr.get_head().author_date]
        for commit in self.commits:
            dates.append(self.gr.get_commit(commit).author_date)
        td=max(dates)-min(dates)
        return td.days
    def calculateBugIntroNum(self):
        bugIntroNum=0
        for commit in self.commits:
            if(commit in self.commitsBug):
                for name in self.commitsBug[commit]["intro"]:
                    if ((name!=self.nameFile) and (".java" in name)):
                        bugIntroNum=bugIntroNum+1
                        break
        return bugIntroNum
    def calculateLogCoupNum(self):
        logCoupNum=0
        for commit in self.commits:
            for modification in self.gr.get_commit(commit).modifications:
                if((not ".java" in modification.filename) or (self.nameFile==modification.filename)):
                    continue
                commitsCoup = self.gr.get_commits_modified_file(modification.filename)
                for commitCoup in commitsCoup:
                    if(self.gr.get_commit(commitCoup).author_date < self.gr.get_commit(commit).author_date):
                        if(commit in self.commitsBug):
                            for name in self.commitsBug[commit]["intro"]:
                                if (name==self.nameFile):
                                    logCoupNum=logCoupNum+1
                                    break
                            else:
                                continue
                            break
                else:
                    continue
                break
        return logCoupNum
    def calculateAvgInterval(self):
        dates=[]
        ddates=[]
        for commit in self.commits:
            dates.append(self.gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod())
        intervalAvg=statistics.mean(ddates)
        return intervalAvg
    def calculateMaxInterval(self):
        dates=[]
        ddates=[]
        for commit in self.commits:
            dates.append(self.gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod())
        intervalMax=max(ddates)
        return intervalMax
    def calculateMinInterval(self):
        dates=[]
        ddates=[]
        for commit in self.commits:
            dates.append(self.gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod())
        intervalMin=min(ddates)
        return intervalMin
    def calculateDevTotal(self):
        developers=[]
        for commit in self.commits:
            developers.append(self.gr.get_commit(commit).author.name)
            developers.append(self.gr.get_commit(commit).committer.name)
            print("author   : "+self.gr.get_commit(commit).author.name)
            print("committer: "+self.gr.get_commit(commit).committer.name)
        return len(set(developers))
    def calculateDevMinor(self):
        devMinor=0
        developers=[]
        setDevelopers=[]
        for commit in self.commits:
            developers.append(self.gr.get_commit(commit).author.name)
            developers.append(self.gr.get_commit(commit).committer.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            if developers.count(developer)/len(developers)<0.2:
                devMinor+=1
        return devMinor
    def calculateDevMajor(self):
        devMajor=0
        developers=[]
        setDevelopers=[]
        for commit in self.commits:
            developers.append(self.gr.get_commit(commit).author.name)
            developers.append(self.gr.get_commit(commit).committer.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            if 0.2<developers.count(developer)/len(developers):
                devMajor+=1
        return devMajor
    def calculateOwnership(self):
        ratio={}
        developers=[]
        setDevelopers=[]
        for commit in self.commits:
            developers.append(self.gr.get_commit(commit).author.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            ratio[developer]=developers.count(developer)/len(developers)
        return ratio[max(ratio)]
    def calculateIsBuggy(self):
        # そのめそっどについてのコミットをその版から遡って、"intro"が最後に来ている→その版でbuggy
        intro=0
        fix=0
        for commit in self.commits:
            if(commit in self.commitsBug):
                if self.nameFile in self.commitsBug[commit]["intro"]:
                    intro=intro+1
                if self.nameFile in self.commitsBug[commit]["fix"]:
                    fix=fix+1
        return intro!=fix
    def calculateHCM(self):
        def calculateH(probabilities):
            sum=0
            for probability in probabilities:
                sum+=(probability*math.log2(probability))
            sum=sum/math.log2(len(probabilities))
            return -sum
        def calculateHCPF(index, probabilities, type):
            print(probabilities)
            if type==3:
                return(1/len(probabilities))*calculateH(probabilities)
            else:
                Exception()