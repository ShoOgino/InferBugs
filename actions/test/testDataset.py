import unittest
from src.dataset import Data, Dataset
from src.utility import Option, UtilPath
from pydriller import GitRepository, RepositoryMining, domain
import os
import json
import csv
class TestDataset(unittest.TestCase):
    def setUp(self):
        name="trial"
        mode="train"
        repositories = [
            {
                "name": "cassandra",
                "url": "https://github.com/apache/cassandra.git",
                "CommitTarget": "df724579efeee15a8974e83be07462a9574b8ae3",
                "filterFile": "",#r".*src\\java\\org\\apache\\cassandra\\service\\WriteResponseHandler.java",
                "codeIssueJira": "CASSANDRA",
                "projectJira": "issues.apache.org/jira",
            }
        ]
        parameters = {}
        option = {
            "name": name,
            "mode": mode,
            "repositories": repositories,
            "parameters": parameters #needless when to infer.
        }
        option = Option(option)

        self.dataset=Dataset(option.getRepositorieImproved())
        self.repository=repositories[0]
        self.gr = GitRepository(os.path.join(UtilPath.Test(), "testDataset/repository"))

    def tearDown(self):
        pass
    def testCalculateLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/loc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateLOC(), int(dataTest[0]))
    def testCalculateAddLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/addLoc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateAddLOC(), int(dataTest[0]))
    def testCalculateDelLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/delLoc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDelLOC(), int(dataTest[0]))
    def testCalculateChgNum(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/chgNum.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                self.assertEqual(Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository)).calculateChgNum(), int(dataTest[0]))
    def testCalculateFixChgNum(self):
        pass
    def testCalculatePastBugNum(self):
        pass
    def testCalculatePeriod(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/period.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculatePeriod(), int(dataTest[0]))
    def testCalculateBugIntroNum(self):
        pass
    def testCalculateLogCoupNum(self):
        pass
    def testCalculateAvgInterval(self):
        pass
    def testCalculateMaxInterval(self):
        pass
    def testCalculateMinInterval(self):
        pass
    def testCalculateDevTotal(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/devTotal.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            print(pathFile)
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDevTotal(), int(dataTest[0]))
    def testCalculateDevMinor(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/devMinor.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                print(pathFile)
                print(data.calculateDevTotal())
                print(data.calculateDevMinor())
                self.assertEqual(data.calculateDevMinor(), int(dataTest[0]))
    def testCalculateDevMajor(self):
        with open(os.path.join(UtilPath.Test(), "testDataset/devMajor.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(),"testDataset/repository/"+dataTest[1]).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDevMajor(), int(dataTest[0]))
    def testCalculateOwnership(self):
        pass
    def testCalculateIsBuggy(self):
        pass
    def testCalculateHCM(self):
        pass

if __name__ == '__main__':
    unittest.main()