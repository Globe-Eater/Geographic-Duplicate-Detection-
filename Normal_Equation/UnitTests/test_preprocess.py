#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:10:35 2019

@author: kellenbullock
"""
import re
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(df):
    """This method takes the target dataframe and preprocesses it into vectors for the Algorithm to handle.
     Will return a CRS sparse matrix for the matrix operations to be calulated on in the Tensorflow graph.
    """
    df = df[['PROPNAME', 'ADDRESS', 'RESNAME']]
    return df

def ngrams(string, n=3):
    string = re.sub(r',-./&',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def vectorize(df):
    """ This takes a dataframe that is of strings only and converts them into vectors.
    The output of this funciton is a crs sparse matrix."""
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_list = []
    for i in df:
        text = df[i]
        tf_idf_list.append(vectorizer.fit_transform(text))
        # Conctenate all spare matrixes
    tf_idf_matrix = hstack([tf_idf_list[0], tf_idf_list[1]]).toarray()
    tf_idf_matrix = hstack([tf_idf_matrix, tf_idf_list[2]]).toarray()
    return tf_idf_matrix

def cosine(tf_idf_matrix):
    ''' X is the input for values in a dataframe for this function. The output is the cosine similarity between
    all X values in the matrix.'''
    similarity_matrix = []
    for i in tf_idf_matrix:
        for x in i:
         	similarity_matrix.append(cosine_similarity(x))
    return similarity_matrix

#tf_idf_matrix = vectorize(df)
#return tf_idf_matrix
#similarity_matrix = cosine(tf_idf_matrix)
#return similarity_matrix

df = pd.read_excel('datasets/prepared_data/Canadian.xlsx')
short_df = preprocess(df)
TF_IDF_vec = vectorize(short_df)
print(TF_IDF_vec[2].shape)
#cosine_vec = cosine(TF_IDF_vec)
#print(cosine_vec[0].shape)
df.shape



