#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:34:39 2019

@author: kellenbullock

Combine all dataframes because I could do in a different file...
"""

import pandas as pd

prop = pd.read_csv('PROPNAME.csv')
prop = prop.drop(columns=['Unnamed: 0'])
res = pd.read_csv('RESNAME.csv')
res = res.drop(columns=['Unnamed: 0'])
add = pd.read_csv('ADDRESS.CSV')
add = add.drop(columns=['Unnamed: 0'])

df = pd.read_excel('Tulsa.xls')

def match_key(dataframe):
    dataframe['left_key'] = ""
    dataframe['right_key'] = ""
    def get_key(col_item):
        for i in col_item:
            answer = df.get_index()
            return answer
  
    dataframe['left_key'] = dataframe['left_key'].apply(match_key)   
    dataframe['right_key'] = dataframe['right_key'].apply(match_key)   
    
match_key(prop)
'''
--------------------------------
First Idea, wasn't working but wanna keep it here
df = pd.read_csv('data.csv')

files = [prop, res, add]

columns = ['PROPNAME', 'RESNAME', 'ADDRESS']

intermediate = df.merge(prop, right_on='left_side', left_on='PROPNAME', left_index=True)
intermediate = intermediate.set_index(['Unnamed: 0'])
#intermediate = intermediate.rename(columns={'Unnamed: 0': 'Index'})

intermediate_2 = intermediate.merge(res, right_on='left_side', left_on='RESNAME', left_index=True)
#intermediate_2 = intermediate_2.set_index(['Unnamed: 0'])

final = intermediate_2.merge(add, right_on='left_side', left_on='RESNAME', left_index=True)

#$final = final.drop_duplicates(subset=final['Index'],keep='first')

final = final.loc[~final.index.duplicated(keep='first')]

final.head()
-----------------------------------
Second idea. Really wasn't working either.

for i in files:
    for x in columns:
        final = df.merge(i, right_on='left_side', left_on=x, left_index=True)
        final.to_csv('final.csv')
       
        #final = final.drop(columns=['left_side','right_side','PROPNAME', 'RESNAME','ADDRESS','Lat','Long'])
        
new = pd.read_csv('final.csv')

'''
'''
#################################
# Third Idea: take the similarity columns out and join the orginial 

prop_sim = prop[['similairity']]
prop_sim = prop_sim.rename(columns={'similairity': 'prop_sim'})

res_sim = res[['similairity']]
res_sim = res_sim.rename(columns={'similairity': 'res_sim'})
'''

'''
------------------------------
Okay my idea now is to do a join between the dataframes and just join the records
share and index togther! With that said I will only be looking at records that have
duplicates in two fields. Rather than some duplicates that have only one field or 
another. 
'''

