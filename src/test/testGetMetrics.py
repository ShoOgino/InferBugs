import unittest

class TestGetMetrics(unittest.TestCase):
    def test_upper(self):
        self.assertEqual(get1(), 1)

if __name__ == '__main__':
    unittest.main()