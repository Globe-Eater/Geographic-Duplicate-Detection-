import pandas as pd
import numpy as np

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
	elif x == 


