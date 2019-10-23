import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidVectorizer
from scipy import hstack
import re



def fill_empty():
	'''fill_empty takes the arugment df (dataframe)
	This should be used with apply.
	For example:
		df.apply(fill_empty)
	'''
	df = df.fillna(Filled_value="No Data")
	return df

def fill_others():
	''' fill_others has no arugments and should be
	used as:
		df = df.apply(fill_others)
	'''
	if x == 0 or '00':
		return 'No Data'
	#elif x == 

def prep(df):
	'''prep is used for vectorizing the data 
	so that it can be used in a machine learning model.

	'''
	def ngrams(string, n=3):
		'''This method both cleans special characters
		out of the strings and prepares themm for being
		proccessed by the TF-IDF'''
		string = re.sub(r',-./&',r'', string)
		ngrams = zip(*[string[i:] for i in range(n)])
		return [''.join(ngram) for ngram in ngrams]

	
