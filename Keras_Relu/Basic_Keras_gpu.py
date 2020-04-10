#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:16:03 2020

@author: kellenbullock
"""

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import pandas as pd
from sklearn.model_selection import train_test_split
from Model_Builder import preprocess
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Load data in:
df = pd.read_excel('/Users/kellenbullock/Desktop/SHPO/Neural_networks/Production/datasets/prepared_data/Oklahoma.xlsx')

# Handle Object ID,
Eval_ObjectID = df['OBJECTID']

# Preprocess the data:
data = preprocess(df)
print('Preprocessing Complete.')

# Train/Test split
m = data
n = df[['duplicate_check']]

x_train, x_test, y_train, y_test = train_test_split(m, n, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=m.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
model.save('/Users/kellenbullock/Desktop/SHPO/Neural_networks/Production/GPU_tests/gpu_model.h5')

