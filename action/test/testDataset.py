import unittest
from src import dataset

class TestDataset(unittest.TestCase):
    def test_upper(self):
        file=dataset.File()
        self.assertEqual(file.get1(), 1)

if __name__ == '__main__':
    unittest.main()