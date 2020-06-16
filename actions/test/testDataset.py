import unittest
from src.dataset import Data, Dataset
from src.utility import Option, UtilPath
from pydriller import GitRepository, RepositoryMining, domain
import os
import json
import csv
class TestDataset(unittest.TestCase):
    def setUp(self):
        name="cassandra20200615"
        mode="train"
        repositories = [
            {
                "name": "cassandra20200615",
                "url": "",
                "CommitTarget": "",
                "filterFile": "",
                "codeIssueJira": "",
                "projectJira": ""
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
        print(os.path.join(UtilPath.Test(), "testDataset",self.repository["name"] ,"repository"))
        self.gr = GitRepository(os.path.join(UtilPath.Test(), "testDataset",self.repository["name"] ,"repository"))

    def tearDown(self):
        pass
    def testCalculateLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"loc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"repository",dataTest[18]).replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateLOC(), int(dataTest[0]))
    def testCalculateAddLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"addLoc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateAddLOC(), int(dataTest[1]))
    def testCalculateDelLOC(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"delLoc.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDelLOC(), int(dataTest[2]))
    def testCalculateChgNum(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"chgNum.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateChgNum(), int(dataTest[3]))
    def testCalculateFixChgNum(self):
        pass
    def testCalculatePastBugNum(self):
        pass
    def testCalculatePeriod(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"period.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculatePeriod(), int(dataTest[11]))
    def testCalculateBugIntroNum(self):
        pass
    def testCalculateLogCoupNum(self):
        pass
    def testCalculateAvgInterval(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"avgInterval.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, dataTest[1].replace("/","\\"), self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateAvgInterval(), int(dataTest[12]))
    def testCalculateMaxInterval(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"maxInterval.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateMaxInterval(), int(dataTest[13]))
    def testCalculateMinInterval(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"minInterval.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateMinInterval(), int(dataTest[14]))
    def testCalculateDevTotal(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"devTotal.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDevTotal(), int(dataTest[7]))
    def testCalculateDevMinor(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"devMinor.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDevMinor(), int(dataTest[8]))
    def testCalculateDevMajor(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"devMajor.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateDevMajor(), int(dataTest[9]))
    def testCalculateOwnership(self):
        with open(os.path.join(UtilPath.Test(), "testDataset", self.repository["name"],"ownership.csv")) as f:
            reader = csv.reader(f)
            datasTest = [row for row in reader][1:]
        for dataTest in datasTest:
            pathFile=dataTest[18].replace("/","\\")
            with self.subTest(pathFile=pathFile):
                data=Data(self.gr, pathFile, self.dataset.getCommitsBug(self.repository))
                self.assertEqual(data.calculateOwnership(), float(dataTest[10]))

    def testCalculateIsBuggy(self):
        pass
    def testCalculateHCM(self):
        pass

if __name__ == '__main__':
    unittest.main()