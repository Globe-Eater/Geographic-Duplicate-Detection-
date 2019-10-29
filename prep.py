import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidVectorizer
from scipy import hstack
import re
from sklearn.model_selection import train_test_split
# For testing purposes will be removed afterward
from paths import pathname

#test line for easy running
df = pd.read_csv(pathname())

def fill_empty(df):
	'''fill_empty takes the arugment df (dataframe)
	This should be used with apply.
	For example:
		df = fill_empty(df)
	'''
	df = df.replace(r'^\s*$', np.nan, regex=True)
	df = df.fillna(Filled_value="No Data")
	return df

def prep(df):
	'''prep is used for vectorizing the data 
	so that it can be used in a machine learning model.
	
	Dev Notes:
	Order of OPS:
		Convert fields like duplicate_check from text to
	1 or 0. 

		Train Test split. 
	'''
	train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
	

	def ngrams(string, n=3):
		'''This method both cleans special characters
		out of the strings and prepares themm for being
		proccessed by the TF-IDF'''
		string = re.sub(r',-./&',r'', string)
		ngrams = zip(*[string[i:] for i in range(n)])
		return [''.join(ngram) for ngram in ngrams]
	
	
	#for x in X_train:
	#Do ngrams things
prep(df)
