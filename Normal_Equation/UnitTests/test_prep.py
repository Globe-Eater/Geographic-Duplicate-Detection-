import unittest
import sys
sys.path.append('/Users/kellenbullock/desktop/shpo/Neural_networks/Production')
import prep
from prep import fill_empty, check_numbers
import pandas as pd

class Testprep(unittest.TestCase):
    '''This testing python program is made for prep.py to ensure all
    train-test-split methods are working as intended and to also make 
    sure all preprocessing is functioning as intended.'''

    def setup(self):
        pass

    def teardown(self):
        pass

    def test_fill_empty(self):
        '''This method is to make sure there are no blank cells left in the dataset.
        We're also going to be looking for NaNs.'''
        data = {
        'ADDRESS': ['Nan', '', 'a$^\s*$', 1, 6],
        'duplicate_check': [1, 0, 2, '', 'Nan']
        }
        df = pd.DataFrame(data=data)
        target = pd.DataFrame(data=[['No Data', 'No Data', 'a', 1, 6], [1, 0, 2, 'No Data', 'No Data']])
        df = fill_empty(df)
        self.assertTrue(target.equals(df))
        
    def test_check_numbers(self):
        '''Check numbers is designed to see if the Lat, and Long points fall within the 
        state boundries of Okalhoma.'''
        df = pd.DataFrame({'Lat': [95.2, 94.3, 97.8], 'Long': [38.2, 32.5, 34.1]})
        target = pd.DataFrame({'Lat': [95.3, 93.1, 97.0],'Long': [39.1, 32.6, 34.1]})
        df = check_numbers(df)
        self.assertEqual(target, df)
        
    def test_labels(self):
        '''This test calls labels in prep.py to test if 1 for poss_dup
        or if 0 for good or if 0 No Data.'''
        df = pd.DataFrame({'duplicate_check': ['good', 'pos_dup', 'No Data', 'pos_dup', 'good']})
        target = pd.DataFrame({'duplicate_check': [0, 1, 0, 1, 0]})
        df['duplicate_check'] = df['duplicate_check'].apply(prep.labels)
        self.assertTrue(target.equals(df))

if __name__ == '__main__':
    unittest.main()
