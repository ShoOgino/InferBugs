import unittest
from src import dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def testCalculateLOC(self):
        file=dataset.File()
        self.assertEqual(file.get1(), 1)
    def testCalculateAddLOC(self):
        pass
    def testCalculateDelLOC(self):
        pass
    def testCalculateChgNum(self):
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