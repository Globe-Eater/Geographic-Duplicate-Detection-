#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:11:11 2019

@author: kellenbullock

This will be the preprocessing file for this project.

This really was a object oriented build of my project but it feel through.

I stil beleive I need a object stored in memonry to feed to other methods. 
"""

import pandas as pd

# columns should be ubiquitous for all data in the SHPO dataset
# These are the ones I believe are worth to partake in the duplicate detection
#columns = ['PROPNAME', 'RESNAME', 'ADDRESS', 'Lat', 'Long', 'duplicate_check']

# Creating a object for the dataframe 
class dataset(object):
    
    def __init__(self, path):
        '''The dataset object will be used to store the path of the data, and 
        the pandas dataframe assocated with the data.'''
        self.path = path
        #self.county = county
    
    
    def prep(self):
        '''This method will:
            read in the data
            Select the columns PROPNAME, RESNAME, ADDRESS, Lat, and Long
            Drop NAN
            Drop Zeros
            
            Then calls for a path for an output denstiation for an intermediate file.'''
        df = pd.read_excel(string)
        # selecting the apporate columns for the dataset
        columns = ['PROPNAME', 'RESNAME', 'ADDRESS', 'Lat', 'Long', 'duplicate_check']
        df = df[columns]
        # Cleaning Nan and 0 values:
        df = df.dropna()
        df = df[(df != 0).all(1)]
        
        def pos_dup(x):
            '''This is for changing the duplicate_check strings into a binary metric
            so that the SVM can proccess targeted data.'''
            if x == 'pos_dup':
                return 1
            elif x == 'good':
                return 0
            else:
                return 'Nan'
        
        df['duplicate_check'] = df['duplicate_check'].apply(pos_dup)
        
        # Request for name of output:
        output = input("Please name and provide the desintation for the file: ")
        
        # Saving output
        df.to_csv(output)
        
        return df
        #df = pd.read_excel("/Users/kellenbullock/Desktop/SHPO/Counties.xls",sheet_name='oklahoma')

# Asking the user for data
# To help the user I will add addtional functionallity to this part.
string = input("Please input the path for the dataset: ")
#county = input('Please input the name of the county: ')

starting = dataset(string)
starting.prep()

print("Preproccessing Finished.")



#


