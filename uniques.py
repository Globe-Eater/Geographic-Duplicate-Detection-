import pandas as pd
import numpy as np

def uniques(df):
	'''I am using this method so I do not have to write df.column_name.uniques
	67 times. 
	Another reason why I want to know this information is to identify the other
	types of No Data. Such as Na or none or 00 ect. 
	
	Example:
		uniques(df)
		returns a dataframe with column names and unique values under each
		column.
	
	Dev Notes:
	.... okay maybe this shouldn't be done on an entire dataset.... 
	Since we have PROPNAME and ADDRESS that are all unique...
	Likely will throw some test cases in later to help prevent this from happening.
	'''
	
	names = []
	for col in df.columns:
		names.append(col)

	uniques = []
	shorten_cols = []
	for i in names:
		x = df[i].unique()
		if len(x) > 50:
			pass
		else:
			uniques.append(x)
			shorten_cols.append(i)
	#Okay I have a good list to work off of, but I dropped all some columns...
	df = pd.DataFrame(uniques,index=shorten_cols)
	df = df.transpose()
	return df

# I am pretty sure this is what I wamt. To get more results increase the len(x) > value.
# Next step is to find out the other No Data type values. such as 00, none, 
#uniques(df)


#if __name__ == '__main__':
#	run()
