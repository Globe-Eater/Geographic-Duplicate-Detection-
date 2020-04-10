#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:15:18 2019

@author: kellenbullock


This will be my attempt at vectorizing the dataset using
sklearn TfidVectorizer and then running sk cosine similarity on that. 

I then will be able to run a SVM algrothim over the data and build my model.

We'll then predict which records will be duplicates with new data and evalute.
"""

import pandas as pd
import numpy as np
#from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in data
df = pd.read_csv('data.csv')
df = df.rename(columns={'Unnamed: 0': 'ObjectId'})

# Clean up nan is duplicate_check
def catcher(dups):
    if dups == '1':
        return 1
    elif dups == '0':
        return 0
    elif dups == 'Nan':
        return 0

df['duplicate_check'] = df['duplicate_check'].apply(catcher)


#df['duplicate_check'] = df['duplicate_check'].apply(catcher)

# Train Test split

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)


# Create ngrams for letters since we are looking at short words not documents
import re

def ngrams(string, n=3):
    string = re.sub(r',-./&',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# list of the columns used for the proccess
columns = ['PROPNAME', 'RESNAME', 'ADDRESS']

#empty dataframes for storage of the different vectors
prop = pd.DataFrame()
res = pd.DataFrame()
add = pd.DataFrame()

dataframes = [prop, res, add]

# storage for the csr matrixs 
matrixs_train = []
# Creating vectors for the training dataset
for i, x in zip(columns, dataframes): 
    x = train_set[i]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    holder = vectorizer.fit_transform(x)
    matrixs_train.append(holder)

matrixs_test = []
# Creating bectors for the test dataset
for i, x in zip(columns, dataframes): 
    x = test_set[i]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    holder = vectorizer.fit_transform(x)
    matrixs_test.append(holder)

# Time to make the Cosine_similarity Matrix
from sklearn.metrics.pairwise import cosine_similarity
# List that will hold the three similarity columns as np arrays for training data:
Metrics_train = []
# creating Cosine_similarity for the training dataset:
for i in matrixs_train:
    temp = cosine_similarity(i)
    Metrics_train.append(temp)
    
# List that will hold the three similarity columns as np sparse arrays for test data:
Metrics_test = []
# Creating the Cosine_sim for the test dataset:
for i in matrixs_test:
    temp = cosine_similarity(i)
    Metrics_test.append(temp)    
    
''' Not real sure if I need to do this or not
This will merge all the sparse dataframes together to get one big dataset
to run through the SVM.
'''
from scipy import hstack

A = Metrics_train[0]
B = Metrics_train[1]
C = Metrics_train[2]

part = hstack([A,B])
full = hstack([part,C])

A = Metrics_test[0]
B = Metrics_test[1]
C = Metrics_test[2]

part_test = hstack([A,B])
full_test = hstack([part_test,C])

# Construct Models

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

X = full
Y = train_set['duplicate_check']


# Poly Kernel Support Vector Machine 
poly_kernel_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C = 1))
        ])

#poly_kernel_svm_clf.fit(X,Y)

# Linear Support Vector Machine
svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C = 1, loss='hinge'))
        ])

#svm_clf.fit(X,Y)   

# Linear Polynomial Support Vector Machine
polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 3)),
        ('scaler', StandardScaler()),
        ('svm_clf', LinearSVC(C = 10, loss='hinge'))
        ])
    
polynomial_svm_clf.fit(X,Y)
    
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X,Y)

# This would work if just using the training dataset. 
def create_single_record():
    single_record = full_test[0]
    single_record = single_record.reshape(1,-1)
    return single_record


# How to evalute the Mean Squared Error on the training set
'''
from sklearn.metrics import mean_squared_error

dups_predicted = poly_kernel_svm_clf.predict(full)
lin_mse = mean_squared_error(train_set['duplicate_check'], dups_predicted)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# result 0.1581550835677955 for the poly kernel 
# result for linear svm:  0.0843149140979014
'''

