#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:45:56 2019

@author: kellenbullock

This is going to be the production grade Neural Network that will proccess 
the whole database.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
# Still working on this:
from prep import *
from paths import pathname

df = pd.read_csv(pathname())
df = df.fillna(value='No Data',axis=1)

df.info()

exit()
# Create Directory for Tensorboard analysis
# Experiement with a better time based format!
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "Production_log"
logdir = "{}/run-{}/".format(root_logdir, now)

# Read in dataset:
df = pd.read_csv() # Path Name

# Split the dataset into training and testing
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

# Preproccessing
df_train = preproccessing(df_train)
df_test = preproccessing(df_test)

labels = None 

# Inputs
training_epochs = 10
learning_rate = 0.01
#hidden_layers = feature_count - 1
cost_history = np.empty(shape=[1], dtype=float)


# Tensorflow constructor:
X = tf.placeholder(tf.float64,[None, feature_count])
Y = tf.placeholder(tf.float64,[None, label_count])
is_training = tf.Variable(True,dtype=tf.bool)



# TensorBoard Summaries
#Independent_variables = tf.summary.scalar(name='Independent_Variables', tensor=X)
#Dependent_variable = tf.summary.scalar(name='Dependent_Variable', tensor=Y)

"""
# TF Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(training_epochs + 1):
"""
