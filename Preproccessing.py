#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:56:43 2019

@author: Kellen Bullock

This individual file is to handle all of the preproccessing
that will occur. There is a lot of text to numeric values that 
needs to be done as well yes/no values that need to converted to
1 or 0. Coded values with letters will also be converted into numeric
types.

The State Historic Presvation Office's mission since 1970 is to 
uphold the laws regarding maintaining National histoical landmarks.

This is done through the National Register Database.
As well as the Oklahoma Landmarks Inventory: the target database of
this machine learning project. 
"""
# This is for testing propuroses and should be removed at time of production
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from scipy import hstack


df = pd.read_excel('Oklahoma.xls')
# ⁨Desktop⁩ ▸ ⁨SHPO⁩ ▸ ⁨Neural_networks⁩
def prep(df):
    """ To be applied to the input DataFrame. This must
    be done before the Tensorflow session is run. On either
    old or new data.
    This is intended for:
        Propname, Address Resname,
        City, Rofftype, Windowtype
        doortype 
        
        DES_RES and , EXTER_FEA,Holds a crap ton of information that will 
        slow down the proccessing speed of the model significantly. I am 
        opting to leave this out for now.
    """
    # Drop all other columns
    df = df[['OBJECTID','PROPNAME','RESNAME','ADDRESS','CITY','ROOF_TYPE',
             'WINDOW_MAT','DOOR_MAT','duplicate_check']]
    
    # Fill NaN
    i = 1
    while 0 < i < 5:
        df.iloc[:, i] = df.iloc[:, i].fillna('No Data')
        i += 1
    
    i = 5
    while 4 < i < 9:
        df.iloc[: , i] = df.iloc[: , i].fillna(0)
        i += 1
        
    
    # Converting PROPNAME, RESNAME, ADDRESS, CITY into TFid Vectors
    def ngrams(string, n=3):
        '''This method both clears certain characters
        out of the strings and prepares them for 
        being proccessed by TF-IDF.'''
        string = re.sub(r',-./&',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]
    # 
    columns = ['PROPNAME', 'RESNAME', 'ADDRESS', 'CITY']
    def vectorize(columns, df):
        """ I need to know the output of this and compare it to 
        the Cosine Similairty results from the testing Alogrothim. 
        
        If just using the TF-IDF is as effective as using the Cosine
        Similarity I will drop that part from this proccess to save time
        and coding heartache. 
        
        As of 9/13/2019 this method does not work.
        
        Upon more thought I am using Cosine similarity to help the 
        neural network better define what are duplicates and what are not.
        Becuase I am giving it numerical value comparisons between other features.
        As of right now I don't think that will hurt it. More testing should be done.
        I did some reading of the research articles that did this method and 
        they all used some sort of metric inorder to make their SVM or whatever
        model run effectively. 
        -9/14/2019
        
        OKay I have a realization that I have to create a separate Dataframe/matrix
        to house the matrixs that will be created from TD-IDF. How this lines up to
        the other two variables I created is a mystery to me. I supppose I concat
        them all together so I can run the alogrithm...?
        -9/15/2019
        """
        
        # I am doing the old way here... :(
        prop = pd.DataFrame()
        res = pd.DataFrame()
        add = pd.DataFrame()
        city = pd.DataFrame()
        
        
        dataframes = [prop, res, add, city]
        
        global matrixs
        matrixs = []
        for i, x in zip(columns, dataframes):
            x = df[columns]
            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
            holder = vectorizer.fit_transform(x)
            matrixs.append(holder)
    # I am calling vectorize later that what I need to.
    # I need to rewrite similarity or make dataframes gobal
    metrics = []
    def similarity(matrixs):
        for i in matrixs:
            temp = cosine_similarity(i)
            metrics.append(temp)
    
    # Execute step 2
    vectorize(columns, df)
    similarity(matrixs)
    
    
    # Step 3: Need to get all the codes for window_typ
    def code_to_number(x):
        '''
        This method is for turning the coded values in
        the columns WINDOW_MAT AND DOOR_MAT into numbers
        so they can be proccessed. 
        
        Originally they had just strings. As seen below in source:
        '''
        if x == 'NO DATA':
            return 0
        elif x == 'NONE LISTED':
            return 1
        elif x == 'EARTH':
            return 2
        elif x == 'WOOD':
            return 3
        elif x == 'Weatherboard':
            return 4
        elif x == 'Shingle':
            return 5
        elif x == 'Log':
            return 6
        elif x == 'Plywood/Particle Board':
            return 7
        elif x == 'Shake':
            return 8
        elif x == 'BRICK':
            return 9
        elif x == 'STONE':
            return 10
        elif x == 'Granite':
            return 11
        elif x == 'Sandstone':
            return 12
        elif x == 'Limestone':
            return 13
        elif x == 'Marble':
            return 14
        elif x == 'Slate':
            return 15
        elif x == 'METAL':
            return 16
        elif x == "Iron":
            return 17
        elif x == 'Copper':
            return 18
        elif x == 'Bronze':
            return 19
        elif x == '99' or 0:
            return 0
    df['WINDOW_MAT'] = df['WINDOW_MAT'].apply(code_to_number)
    df['DOOR_MAT'] = df['DOOR_MAT'].apply(code_to_number)
    
    def dup_check_to_number(x):
        if x == 'good':
            return 0
        elif x == 'pos_dup':
            return 1
	else:
	    return 0
    df['duplicate_check'] = df['duplicate_check'].apply(dup_check_to_number)
    #This returns the dataframe to the __main__ program
    return df
# Make sure you remove this once Prep is complete!!!!
prep(df)
