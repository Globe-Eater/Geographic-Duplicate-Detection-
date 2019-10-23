import pandas as pd
import numpy as np

df = pd.read_excel('Oklahoma.xls')
# There's a lot more columns to drop..... like vicnity 
#df = df.drop(columns=['PROPNAME','RESNAME','ADDRESS','Lat','Long','duplicate_check'])
df = df[['HIST_FUNC','CURR_FUNC','AREASG_1','AREASG_2']]

def uniques(df):
	'''I am using this method so I do not have to write df.column_name.uniques
	67 times. 
	Another reason why I want to know this information is to identify the other
	types of No Data. Such as Na or none or 00 ect. 
	
	Example:
		uniques(df)
		returns a dataframe with column names and unique values under each
		column.

	.... okay maybe this shouldn't be done on an entire dataset.... 
	Since we have PROPNAME and ADDRESS that are all unique...
	'''
	
	names = []
	for col in df.columns:
		print(col)
		names.append(col)

	uniques = []
	for i in names:
		x = df[i].unique()
		uniques.append(x)
	
	df = pd.DataFrame(uniques,index=names)
	df = df.transpose()
	print(df)

uniques(df)

#if __name__ == '__main__':
#	run()
