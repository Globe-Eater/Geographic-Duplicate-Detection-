#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:06:51 2020

@author: kellenbullock
"""
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from Model_Builder import preprocess

# Load data in:
df = pd.read_excel('/Users/kellenbullock/Desktop/SHPO/Neural_networks/Production/datasets/prepared_data/Garfield.xlsx')
df1 = pd.read_excel('/Users/kellenbullock/Desktop/SHPO/Neural_networks/Production/datasets/prepared_data/Oklahoma.xlsx')

# Handle Object ID,
Eval_ObjectID = df['OBJECTID']

# Preprocess the data:
data = preprocess(df)
print('Preprocessing Complete.')

# Train/Test split
m = data
n = df[['duplicate_check']]

x_train, x_test, y_train, y_test = train_test_split(m, n, test_size=0.2, random_state=42)

model = load_model('/Users/kellenbullock/Desktop/SHPO/Neural_networks/Production/GPU_tests/gpu_model.h5')

classes = model.predict(x_test, batch_size=128)