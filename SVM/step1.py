#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:46:16 2019

@author: kellenbullock

PREPROCCESSING part

This is mark 2 of preproccessing. I am combining more steps into one defintion that way
I only have to run a method once rather than do all the functions of pandas spearately. 

This part reads in the data into a pandas dataframe. Then selects the data most pertnate
to the study and then displays the data.
"""

import pandas as pd

# will need to add county selection here later on. As well as a try, expect block.
string = input("Please input path name: ")

# creation of dataframe object
df = pd.read_excel(string)

# selecting the apporate columns for the dataset
columns = ['PROPNAME', 'RESNAME', 'ADDRESS', 'Lat', 'Long']
df = df[columns]

# Cleaning Nan and 0 values:
df = df.dropna()
df = df[(df != 0).all(1)]

# Request for name of output:
output = input("Please name and provide the desintation for the file: ")

# Saving output
df.to_csv(output)