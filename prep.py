import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidVectorizer
from scipy import hstack
import re
from sklearn.model_selection import train_test_split
import re
import pandas as pd

def fill_empty(df):
    '''fill_empty takes the arugment df (dataframe)
    This should be used with apply.
    For example:
        df = fill_empty(df)
    '''
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(Filled_value="No Data")
    assert df , 'DataFrame has not loaded in try again.'
    return df

def checkNumbers(lat, long):
    '''This method is to make sure that the lats and longs are within the state of Oklahoma.
    inputs: df['Lat', 'long']]
    output "Everything should be within Oklahoma.
    output There is a value that is outside the range of Oklahoma.'''
    # assert # are the numbers within Oklahoma? Need to look this up and impliment it.
    pass

def prep(df):
    '''prep is used for vectorizing the data 
    so that it can be used in a machine learning model.
    Dev Notes:
    Order of OPS:
    Convert fields like duplicate_check from text to
    1 or 0. 
    Train Test split. 
    '''
    df = df[['OBJECTID', 'ADDRESS', 'RESNAME', 'Lat', 'Long', 'duplicate_check']]
    def labels(duplicate_check):
        '''This method is to be applied to the dataframe df, that has a column duplicate_check in it.
        This is being used to convert poss_dup to 1 or good to 0. Later this will be expanded to cover anything else
        within the dataset.'''
        if duplicate_check == 'good':
            return 0
        else:
            return 1
