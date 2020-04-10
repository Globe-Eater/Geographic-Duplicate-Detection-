#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:04:34 2019

@author: kellenbullock

This section will be for generating:
    > Metrics for character differences
    > machine learning outputs:
        SVM
        Logistic
        Nerual networks
        
   Aug 14, 2019 This is an attempt to get SVM to work:
       
        
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

prop_test = pd.read_csv('Prop_test.csv')
prop_test = prop_test.drop(columns=['Unnamed: 0'])
X = prop_test['similairity']
Y = prop_test['duplicate_check']

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearSVC(C=1)),
        ])

svm_clf.fit(X,Y)

