#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:14:55 2020

@author: kellenbullock
"""

import pandas as pd
import numpy as np
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" # Deleteing the first @ will use the gpu if configured prior.
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
from sklearn.metrics import classification_report, confusion_matrix

path = '/Users/kellenbullock/desktop/SHPO/OLI.xls'
df = pd.read_excel(path)

df = df[['OBJECTID','PROPNAME','COUNTYCD', 'RESNAME', 'ADDRESS', 'Lat', 'Long', 'duplicate_check']]

def convert_labels(labels):
    '''This method changes all labels to 1 or 0.
    Usage: 
         df['duplicate_check'] = df['duplicate_check'].apply(convert_labels)
    If there is an error will return Label Problem.'''
    if labels == 'good':
        return 1
    elif labels == 'pos_dup':
        return 0
    elif labels == 'Not a duplicate':
        return 1
    elif labels == 'Requires more inspection':
        return 0
    elif labels == 'Duplicate':
        return 0
    elif labels == 'pgood':
        return 1
    elif labels == 'pos_dup_':
        return 0
    elif labels == 'duplicate':
        return 0
    
    assert labels == 1 or 0, "Label Problem, check data for anything other than good or pos_dup."
# dropping rows without labels. This will likely be a test set of some varitiy. 
df = df.dropna(subset=['duplicate_check'])
df['duplicate_check'] = df['duplicate_check'].apply(convert_labels)

dups = df[df['duplicate_check'] == 1]

df = df.dropna(subset=['Lat', 'Long'])

def null_breaker(df, column):
    for column in df:
        df[column] = df[column].fillna(value='NO DATA')
        df[column] = df[column].replace('', "NO DATA")
        df[column] = df[column].replace(' ', "NO DATA")
        #df[column] = df[column].str.upper()

def no_data(x):
    if x == "99 UNCOLLECTED":
        return "NO DATA"
    elif x == None:
        return "NO DATA"
    elif x == "99 UNCOLLECTED":
        return "NO DATA"
    elif x == "none":
        return "NO DATA"
    else:
        return x

def cleaner(df):
    df['Lat'] = df['Lat'].astype('double')
    df['Long'] = df['Long'].astype('double')
    for column in df:
        try:
            if df[column].dtypes == float:
                df[column] = df[column].astype('double')
        except:
            if df[column].dtypes == object:
                df[column] = df[column].apply(no_data)
                df[column] = df[column].str.lower()
                null_breaker(df, "BLOCK")
                
        #elif df[column].dtypes == float:
        #    df[column] = df[column].astype('double')
            else:
                print("Skipping, because this is an datetime or int type.")
            
    return df

df = cleaner(df)

def tokenize(df):
    '''This method will turn the fields PROPNAME, RESNAME, and ADDRESS into vectors for the machine learning model:
    inputs:
        A pandas dataframe of the Oklahoma landmarks inventory data. This can be from a online database submit form
        or a csv/excel copy of the database.
    outputs:
        A spare matrix array of vectors.
        
    Usage:
        spare_matrix_variable_name = tokenize(df)'''
    
    df = df[['OBJECTID','PROPNAME','COUNTYCD', 'RESNAME', 'ADDRESS', 'Lat', 'Long', 'duplicate_check']]
    propname = df['PROPNAME'].astype(str)
    address = df['ADDRESS'].astype(str)
    resname = df['RESNAME'].astype(str)
    tokenize = Tokenizer(num_words=3000)
    tokenize.fit_on_texts(propname)
    tokenize.fit_on_texts(address)
    tokenize.fit_on_texts(resname)

    x_data = tokenize.texts_to_matrix(propname)
    y_data = tokenize.texts_to_matrix(address)
    z_data = tokenize.texts_to_matrix(resname)

    #print(x_data.shape)
    #print(y_data.shape)
    #print(z_data.shape)

    doneso = np.column_stack((x_data, y_data))
    doneso = np.column_stack((doneso, z_data))
    latso = df['Lat'].values
    longso = df['Long'].values
    doneso = np.column_stack((doneso, latso))
    doneso = np.column_stack((doneso, longso))
    return doneso

indepdents = tokenize(df)

# Train/Test split
m = indepdents
n = df[['duplicate_check']]

x_train, x_test, y_train, y_test = train_test_split(m, n, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=m.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, validation_split=0.25, epochs=180, batch_size=256, verbose=1)

score = model.evaluate(x_test, y_test, batch_size=64)
model.save('/users/kellenbullock/desktop/shpo/SAVED_MODELS/Propname_Address_LOCATION_model.h5')
print(score)
end = time.time()
print("time to run: ", end - start)

