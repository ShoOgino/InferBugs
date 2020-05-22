import unittest
from src.dataset import Metrics
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
        pass
    def testCalculateAddLOC(self):
        pass
    def testCalculateDelLOC(self):
        pass
    def testCalculateChgNum(self):
        for key in self.df:
            pathFile=os.path.abspath("../dataset/trial/cassandra/cassandra/"+key).replace("\\","/")
            with self.subTest(pathFile=pathFile):
                nameFile=os.path.basename(pathFile)
                commits =[]
                for commit in self.gr.get_commits_modified_file(os.path.abspath(pathFile)):
                    for modification in self.gr.get_commit(commit).modifications:
                        if (nameFile == modification.filename):
                            print(modification.filename)
                            print(modification.change_type)
                            commits.append(commit)
                            if (modification.change_type!=modification.change_type.MODIFY) and (modification.change_type!=modification.change_type.DELETE):
                                break
                    else:
                        continue
                    break
                self.assertEqual(
                    str(
                        Metrics.calculateChgNum(
                            self.gr,
                            commits,
                            nameFile)),
                    self.df[key]["chgNum"])
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