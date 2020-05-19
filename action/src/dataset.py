import glob
import os
import sys
from pydriller import GitRepository, RepositoryMining, domain
import csv
import json
import re
from urllib.parse import quote
import urllib.request as url
import math
import statistics
from tqdm import tqdm

import subprocess
def local_chdir(func):
    def _inner(*args, **kwargs):
        dir_original = os.getcwd()
        ret = func(*args, **kwargs)
        os.chdir(dir_original)
        return ret
    return _inner
def cloneRepository(urlRepository, pathRepository):
    cmdGitClone=["git", "clone"]
    cmdGitClone.extend([urlRepository, pathRepository])
    subprocess.call()
@local_chdir
def checkoutRepository(pathRepository, IdCommit):
    os.chdir(pathRepository)
    cmdGitCheckout=["git", "checkout"]
    cmdGitCheckout.extend([IdCommit])

class Dataset:
    def __init__(self,urlRepository, pathData, nameRepository, filterFile, codeIssueJira, nameProjectJira):
        self.data=[]
        self.urlRepository=urlRepository
        self.pathData=pathData
        self.nameRepository=nameRepository
        self.pathRepository=os.path.join(self.pathData, nameRepository)
        self.filterFile=filterFile
        self.codeIssueJira=codeIssueJira
        self.nameProjectJira=nameProjectJira
        #cloneRepository(urlRepository, pathRepository)
        #checkoutRepository(pathRepository, identifierDataset.IdCommit)
        self.gr = GitRepository(self.pathRepository)

    def calculateDataset(self):
        pathsFile=glob.glob(self.pathRepository+"/**/*.java", recursive=True)
        json_open=open(os.path.join(self.pathData, "annotationsReformatted.json"), 'r')
        formattedAnnotations=json.load(json_open)

        with tqdm(pathsFile,bar_format="{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]") as pbar:
            for pathFile in pbar:
                nameFile=os.path.basename(pathFile)
                commits = self.gr.get_commits_modified_file(os.path.abspath(pathFile))
                print(pathFile)
                self.data.append([
                    UtilityMetrics.calculateLOC(pathFile),
                    UtilityMetrics.calculateAddLOC(self.gr,commits,nameFile),
                    UtilityMetrics.calculateDelLOC(self.gr,commits,nameFile),
                    UtilityMetrics.calculateChgNum(self.gr,commits,nameFile),
                    UtilityMetrics.calculateFixChgNum(self.gr, commits, nameFile),
                    UtilityMetrics.calculatePastBugNum(self.gr, commits, nameFile, formattedAnnotations),
                    UtilityMetrics.calculateHCM(self.gr, commits, nameFile,formattedAnnotations),
                    UtilityMetrics.calculateDevTotal(self.gr,commits,nameFile),
                    UtilityMetrics.calculateDevMinor(self.gr,commits,nameFile),
                    UtilityMetrics.calculateDevMajor(self.gr,commits,nameFile),
                    UtilityMetrics.calculateOwnership(self.gr,commits,nameFile),
                    UtilityMetrics.calculatePeriod(self.gr,commits,nameFile),
                    UtilityMetrics.calculateAvgInterval(self.gr,commits,nameFile),
                    UtilityMetrics.calculateMaxInterval(self.gr,commits,nameFile),
                    UtilityMetrics.calculateMinInterval(self.gr,commits,nameFile),
                    UtilityMetrics.calculateLogCoupNum(self.gr, commits, nameFile, formattedAnnotations),
                    UtilityMetrics.calculateBugIntroNum(self.gr,commits,nameFile,formattedAnnotations),
                    UtilityMetrics.calculateIsBuggy(self.gr,commits,nameFile, formattedAnnotations),
                    pathFile
                ]
                )


class UtilityMetrics:
    @staticmethod
    def get1():
        return 1
    @staticmethod
    def calculateLOC(pathFile):
        with open(pathFile, "r") as fr:
            return len(fr.readlines())
    @staticmethod
    def calculateAddLOC(gr,commits,nameFile):
        addLOC=0
        for commit in commits:
            for modification in gr.get_commit(commit).modifications:
                if modification.filename==nameFile:
                    addLOC+=modification.added
        return addLOC
    @staticmethod
    def calculateDelLOC(gr,commits,nameFile):
        delLOC=0
        for commit in commits:
            for modification in gr.get_commit(commit).modifications:
                if modification.filename==nameFile:
                    delLOC+=modification.removed
        return delLOC
    @staticmethod
    def calculateChgNum(gr,commits,nameFile):
        return len(commits)
    @staticmethod
    def calculateFixChgNum(gr, commits, nameFile):
        fixChgNum=0
        gitlog_pattern= r'(?<=CASSANDRA-)\d+|(?<=#)\d+'
        json_open = open('../dataset/trial/cassandra/IDsBug.json', 'r')
        IDsBug = json.load(json_open)["IDsBug"]
        for commit in commits:
            result=re.match(gitlog_pattern, gr.get_commit(commit).msg)
            if(result):
                if(result.group() in IDsBug):
                    fixChgNum=fixChgNum+1
                    break
        return fixChgNum
    @staticmethod
    def calculatePastBugNum(gr,commits, nameFile,formattedAnnotations):
        pastBugNum=0
        for commit in commits:
            if(commit in formattedAnnotations):
                if(nameFile in formattedAnnotations[commit]["fix"]):
                    pastBugNum=pastBugNum+1
        return pastBugNum
    @staticmethod
    def calculatePeriod(gr, commits, nameFile):
        dates=[gr.get_head().author_date]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        td=max(dates)-min(dates)
        return td.days
    @staticmethod
    def calculateBugIntroNum(gr, commits, nameFile, formattedAnnotations):
        bugIntroNum=0
        for commit in commits:
            if(commit in formattedAnnotations):
                for name in formattedAnnotations[commit]["intro"]:
                    if ((name!=nameFile) and (".java" in name)):
                        bugIntroNum=bugIntroNum+1
                        break
        return bugIntroNum
    @staticmethod
    def calculateLogCoupNum(gr, commits, nameFile, formattedAnnotations):
        logCoupNum=0
        for commit in commits:
            for modification in gr.get_commit(commit).modifications:
                if((not ".java" in modification.filename) or (nameFile==modification.filename)):
                    continue
                commitsCoup = gr.get_commits_modified_file(modification.filename)
                for commitCoup in commitsCoup:
                    if(gr.get_commit(commitCoup).author_date < gr.get_commit(commit).author_date):
                        if(commit in formattedAnnotations):
                            for name in formattedAnnotations[commit]["intro"]:
                                if (name==nameFile):
                                    logCoupNum=logCoupNum+1
                                    break
                            else:
                                continue
                            break
                else:
                    continue
                break
        return logCoupNum
    @staticmethod
    def calculateAvgInterval(gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(UtilityMetrics.calculatePeriod(gr, commits, nameFile))
        intervalAvg=statistics.mean(ddates)
        return intervalAvg
    @staticmethod
    def calculateMaxInterval(gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(UtilityMetrics.calculatePeriod(gr, commits, nameFile))
        intervalMax=max(ddates)
        return intervalMax
    @staticmethod
    def calculateMinInterval(gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(UtilityMetrics.calculatePeriod(gr, commits, nameFile))
        intervalMin=min(ddates)
        return intervalMin
    @staticmethod
    def calculateDevTotal(gr,commits,nameFile):
        developers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        return len(set(developers))
    @staticmethod
    def calculateDevMinor(gr,commits,nameFile):
        devMinor=0
        developers=[]
        setDevelopers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            if developers.count(developer)/len(developers)<0.2:
                devMinor+=1
        return devMinor
    @staticmethod
    def calculateDevMajor(gr,commits,nameFile):
        devMajor=0
        developers=[]
        setDevelopers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            if 0.2<developers.count(developer)/len(developers):
                devMajor+=1
        return devMajor
    @staticmethod
    def calculateOwnership(gr,commits,nameFile):
        ratio={}
        developers=[]
        setDevelopers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            ratio[developer]=developers.count(developer)/len(developers)
        return ratio[max(ratio)]
    @staticmethod
    def calculateIsBuggy(gr,commits,nameFile, formattedAnnotations):
        # そのめそっどについてのコミットをその版から遡って、"intro"が最後に来ている→その版でbuggy
        intro=0
        fix=0
        for commit in commits:
            if(commit in formattedAnnotations):
                if nameFile in formattedAnnotations[commit]["intro"]:
                    intro=intro+1
                if nameFile in formattedAnnotations[commit]["fix"]:
                    fix=fix+1
        return intro!=fix
    @staticmethod
    def calculateHCM(gr,commits,nameFile, formattedAnnotations):
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
def reformatAnnotations():
    annotationsReformatted={}
    pathRepo=sys.argv[1]
    json_open = open(sys.argv[2], 'r')
    json_load = json.load(json_open)
    for commit in json_load:
        for i in range(len(json_load[commit])):
            if(not commit in annotationsReformatted):
                annotationsReformatted[commit]={"fix":[], "intro":[]}
            annotationsReformatted[commit]["fix"].append(json_load[commit][i]["filePath"])
            for revision in json_load[commit][i]["revisions"]:
                if(revision!=commit):
                    if(not revision in annotationsReformatted):
                         annotationsReformatted[revision]={"fix":[], "intro":[]}
                    annotationsReformatted[revision]["intro"].append(json_load[commit][i]["filePath"])
    with open('annotationsReformatted.json', 'w') as f:
        json.dump(annotationsReformatted, f, indent=4)
def fetchIdsBug(project_issue_code, jira_project_name):
    IDsBug=[]
    jql = 'project = ' + project_issue_code + ' ' \
        + 'AND issuetype = Bug '\
        + 'AND status in (Resolved, Closed) '\
        + 'AND resolution = Fixed '
    jql = quote(jql, safe='')
    start_at = 0
    max_results = 1000
    request = 'https://' + jira_project_name + '/rest/api/2/search?'\
        + 'jql={}&startAt={}&maxResults={}'
    with url.urlopen(request.format(jql, start_at, '1')) as conn:
        contents = json.loads(conn.read().decode('utf-8'))
        total = contents['total']
    while start_at < total:
        with url.urlopen(request.format(jql, start_at, max_results)) as conn:
            issues=json.loads(conn.read().decode('utf-8'))
            for issue in issues["issues"]:
                IDsBug.append(issue["key"][10:])
        start_at += max_results
    d={"IDsBug":IDsBug}
    with open('IDsBug.json', 'w') as f:
        json.dump(d, f, indent=4)