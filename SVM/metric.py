#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:40:46 2019

@author: kellenbullock

This file will be used to figure out the score metric for the SVM.
"""

import pandas as pd
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

df = pd.read_excel('Tulsa.xls')
df = df.dropna()
#df = df.drop(columns=['Unnamed: 0'])
'''
# This section is an explority study of Fuzzywuzzy Levenstine distance between strings.

#holder = df.loc[lambda df: df['CITY'] == 'ARDMORE']


choices = df['RESNAME'].unique()
print(choices)


pookie = df[df['RESNAME'].str.contains("6.*DAKOTA", na=False)]['RESNAME'].value_counts()
print(pookie)


done = process.extract("CALUMET FIRST NATIONAL BANK", choices, limit=30, scorer=fuzz.token_sort_ratio)
print(done)

#____________________________________
#Okay this is how I think it is going to look:
    
def find_unquines():
   use the fuzz.unquine() method and save to a list
    list_1 = df['PROPNAME'].unique()
    list_2 = df['RESNAME'].unique()
    
    prop_done = process.extract(list_1, limit = 10, scorer=fuzz.token_sort_ratio)
    res_done = process.extract(list_2, limit = 10, socorer=fuzz.token_sort_ratio)
    
    #Turn these into a dataframe with lat and log attatched.
    



______________________________________

This part is for the loop for the cosine metric:
    
How this file should have been written:
    columns = ['PROPNAME', 'RESNAME','ADDRESS']
    
for i in columns:
    from sklearn.feature_extraction.text import TfidfVectorizer
    company_names = df[i]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(company_names)
    
        
    import time
    t1 = time.time()
    matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.8)
    t = time.time()-t1
    print("SELFTIMED:", t)
    
    
    matches_df = get_matches_df(matches, company_names, top=9590)
    matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
    matches_df.sample(20)
    
    matches_df.to_csv(i+'.csv')



-------------------------------------------------
Cosine Metric
'''
columns = ['PROPNAME', 'RESNAME','ADDRESS']

import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))



import re

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]



from sklearn.feature_extraction.text import TfidfVectorizer
company_names = df['PROPNAME']
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(company_names)


import time
t1 = time.time()
matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.8)
t = time.time()-t1
print("SELFTIMED:", t)


def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
     return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})


    
matches_df = get_matches_df(matches, company_names, top=9590)
matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
matches_df.sample(20)


