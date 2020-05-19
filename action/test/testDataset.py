import unittest
from src.dataset import UtilityMetrics
from pydriller import GitRepository, RepositoryMining, domain
import os
import json
class TestDataset(unittest.TestCase):
    def setUp(self):
        self.pathRepository="C:/Users/login/work/InferBugs/dataset/trial/cassandra/cassandra"
        self.pathDataCorrect="test/metrics.json"
        self.gr = GitRepository(self.pathRepository)
        with open(self.pathDataCorrect) as f:
            self.df = json.load(f)
    def tearDown(self):
        pass

    def testCalculateLOC(self):
        self.assertEqual(UtilityMetrics.get1(), 1)
    def testCalculateAddLOC(self):
        pass
    def testCalculateDelLOC(self):
        pass
    def testCalculateChgNum(self):
        for pathFile in self.df:
            pathFile1=os.path.abspath("../dataset/trial/cassandra/cassandra/"+pathFile).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                nameFile=os.path.basename(pathFile)
                print(nameFile)
                self.assertEqual(str(UtilityMetrics.calculateChgNum(self.gr,self.gr.get_commits_modified_file(os.path.join(self.pathRepository,pathFile1)),nameFile)),self.df[pathFile]["chgNum"])
        pass
    def testCalculateFixChgNum(self):
        pass
    def testCalculatePastBugNum(self):
        pass
    def testCalculatePeriod(self):
        pass
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
        pass
    def testCalculateDevMinor(self):
        pass
    def testCalculateDevMajor(self):
        pass
    def testCalculateOwnership(self):
        pass
    def testCalculateIsBuggy(self):
        pass
    def testCalculateHCM(self):
        pass

if __name__ == '__main__':
    unittest.main()