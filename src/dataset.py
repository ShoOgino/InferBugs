import glob
import os
import sys
from pydriller import GitRepository, RepositoryMining, domain
import time
from tqdm import tqdm
import math
import statistics

import csv

import pandas as pd
import json

from urllib.parse import quote
import urllib.request as url
import json
import argparse
import io

import re
import time
import datetime

class File:
    def get1(self):
        return 1
    def calculateLOC(self, pathFile):
        with open(pathFile, "r") as fr:
            return len(fr.readlines())
    def calculateAddLOC(self, gr,commits,nameFile):
        addLOC=0
        for commit in commits:
            for modification in gr.get_commit(commit).modifications:
                if modification.filename==nameFile:
                    addLOC+=modification.added
        return addLOC
    def calculateDelLOC(self, gr,commits,nameFile):
        delLOC=0
        for commit in commits:
            for modification in gr.get_commit(commit).modifications:
                if modification.filename==nameFile:
                    delLOC+=modification.removed
        return delLOC
    def calculateChgNum(self, gr,commits,nameFile):
        return len(commits)
    def calculateFixChgNum(self, gr, commits, nameFile):
        fixChgNum=0
        gitlog_pattern= r'(?<=CASSANDRA-)\d+|(?<=#)\d+'
        json_open = open('IDsBug.json', 'r')
        IDsBug = json.load(json_open)["IDsBug"]
        for commit in commits:
            result=re.match(gitlog_pattern, gr.get_commit(commit).msg)
            if(result):
                if(result.group() in IDsBug):
                    fixChgNum=fixChgNum+1
                    break
        return fixChgNum
    def calculatePastBugNum(self, gr,commits, nameFile,formattedAnnotations):
        pastBugNum=0
        for commit in commits:
            if(commit in formattedAnnotations):
                if(nameFile in formattedAnnotations[commit]["fix"]):
                    pastBugNum=pastBugNum+1
        return pastBugNum
    def calculatePeriod(self, gr, commits, nameFile):
        dates=[gr.get_head().author_date]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        td=max(dates)-min(dates)
        return td.days
    def calculateBugIntroNum(self, gr, commits, nameFile, formattedAnnotations):
        bugIntroNum=0
        for commit in commits:
            if(commit in formattedAnnotations):
                for name in formattedAnnotations[commit]["intro"]:
                    if ((name!=nameFile) and (".java" in name)):
                        bugIntroNum=bugIntroNum+1
                        break
        return bugIntroNum
    def calculateLogCoupNum(self, gr, commits, nameFile, formattedAnnotations):
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
    def calculateAvgInterval(self, gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod(gr, commits, nameFile))
        intervalAvg=statistics.mean(ddates)
        return intervalAvg
    def calculateMaxInterval(self, gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod(gr, commits, nameFile))
        intervalMax=max(ddates)
        return intervalMax
    def calculateMinInterval(self, gr,commits,nameFile):
        dates=[]
        ddates=[]
        for commit in commits:
            dates.append(gr.get_commit(commit).author_date)
        dates.sort()
        for i in range(len(dates)-1):
            ddates.append((dates[i+1]-dates[i]).days)
        if len(ddates)==0:
            ddates.append(self.calculatePeriod(gr, commits, nameFile))
        intervalMin=min(ddates)
        return intervalMin
    def calculateDevTotal(self, gr,commits,nameFile):
        developers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        return len(set(developers))
    def calculateDevMinor(self, gr,commits,nameFile):
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
    def calculateDevMajor(self, gr,commits,nameFile):
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
    def calculateOwnership(self, gr,commits,nameFile):
        ratio={}
        developers=[]
        setDevelopers=[]
        for commit in commits:
            developers.append(gr.get_commit(commit).author.name)
        setDevelopers=set(developers)
        for developer in setDevelopers:
            ratio[developer]=developers.count(developer)/len(developers)
        return ratio[max(ratio)]
    def calculateIsBuggy(self, gr,commits,nameFile, formattedAnnotations):
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
    def calculateHCM(self, gr,commits,nameFile, formattedAnnotations):
        def calculateH(probabilities):
            sum=0
            for probability in probabilities:
                sum+=(probability*math.log2(probability))
            sum=sum/math.log2(len(probabilities))
            return -sum
        def calculateHCPF(self, index, probabilities, type):
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