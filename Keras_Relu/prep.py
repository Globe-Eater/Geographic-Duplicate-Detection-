import numpy as np
import pandas as pd


def start():
    '''This method is designed to read input in from the user.'''
    df = pd.read_excel("datasets/unprepared_data/" + input("Please enter the path for the data:"))
    return df

def fill_empty(df):
    '''fill_empty takes the arugment df (dataframe)
    This should be used with apply.
    For example:
        df = fill_empty(df)'''
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(value="No Data")
    return df

def check_Lat(x):
    '''This method is to make sure that the lats and longs are within the state of Oklahoma.
    inputs: df['Lat', 'long']]
    output "Everything should be within Oklahoma.
    output There is a value that is outside the range of Oklahoma.'''
    # assert # are the numbers within Oklahoma? Need to look this up and impliment it.
    if x < 97.8:
        assert False, 'There is a point too far East.'
    elif x > 103:
        assert False, 'There is a point too far West.'
        
def check_long(y):
    '''This method is designed to ensure that the Longitude for the points is within OK.'''
    if y > 37:
        assert False, "There is a point that is too far North."
    elif y < 34.5:
        assert False, "There is a point that is too far South."

def prep(df):
    '''prep is used for vectorizing the data so that it can be used
    in a machine learning model.
    Dev Notes:
    Order of OPS:
    Convert fields like duplicate_check from text to
    1 or 0.'''
    df = df[['OBJECTID', 'PROPNAME', 'ADDRESS', 'RESNAME', 'Lat', 'Long', 'duplicate_check']]
    return df

def labels(x):
    '''This method is to be applied to the dataframe df, that has a column duplicate_check in it.
    This is being used to convert poss_dup to 1 or good to 0. Later this will be expanded to cover
    anything else within the dataset.'''
    if x == 'pos_dup':
        return 1
    elif x == 'good':
        return 0
    elif x == 'No Data':
        return 0
    else:
        return 0

def saver(df):
    '''This method is designed to ask the user if they want to save and if so where.
    The arguments are for asking the user if they want to save, and the dataframe to
    be saved.'''
    user_input = input("Would you like to save y/n?: ")
    question = True
    while question:
        if user_input == 'n':
            break
        elif user_input == 'y':
            path = input('Please input a valid path and filename such as /path/to/file/.xlsx : ')
            try:
                df.to_excel(path)
                print("File successfully saved.")
                question = False
            except FileNotFoundError:
                print("Path was not found please try again")

if __name__ == '__main__':
    dataframe = start()
    dataframe = fill_empty(dataframe)
    dataframe = prep(dataframe)
    dataframe['duplicate_check'] = dataframe['duplicate_check'].apply(labels)
    dataframe.head()
    saver(dataframe)
