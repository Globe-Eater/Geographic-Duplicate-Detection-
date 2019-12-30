import unittest
import sys
sys.path.append('/Users/kellenbullock/desktop/shpo/Neural_networks/Production')
import prep
import pandas as pd

class Testprep(unittest.TestCase):
    '''This testing python program is made for prep.py to ensure all train-test-split methods are working as intended
    and to also make sure all preprocessing is functioning as intended.'''

    def setup(self):
        pass

    def teardown(self):
        pass

    def test_fill_empty(self):
        pass

    def test_checknumbers(self):
        pass

    def test_prep(self):
        pass

    def test_labels(self):
        '''This test calls labels in prep.py to test if 1 for poss_dup
        or if 0 for good or if 0 No Data.'''
        1 = 'pos_dup'
        0 = 'good'
        0 = 'No Data'
        

if __name__ == '__main__':
    unittest.main()
