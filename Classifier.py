import tensorflow as tf
import numpy as np
import pandas as pd
from prep import start
from Model_Builder import preprocess
from sklearn.model_selection import train_test_split

#def main():
'''This program will predict duplicate records that are input into it.
Requires: prepared data from prep.py.

Output: an excel file of objectID and prediction.
'''

# Load data in:
df = start()

# Handle Object ID,
Eval_ObjectID = df['OBJECTID']

# Conver to vectors
data = preprocess(df)

# Shapes of Features (independent variables) and Labels (dependent variables)
feature_count = data.shape[0]
label_count = df['duplicate_check'].shape[0]

# Conver to vectors
data = preprocess(data)

# Here I am going to reverse the dimensions of X_train as a new variable:
data_plus_bias = np.c_[np.ones((feature_count, 1)), data]
data_plus_bias = np.transpose(data_plus_bias)

# Train_set, Testing_Set split:
X = data
y = data[['duplicate_check']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

n_epochs = 10 # This will absoutely be played with during testing.
learning_rate = 0.01 # This value is set to low inorder to make sure the algorithm decends the gradient.

X = tf.placeholder(tf.float32, shape=(None, label_count), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
theta = tf.Variable(tf.random_uniform([label_count, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

# This loads the graph structure
saver = tf.train.import_meta_graph("saved_models/model_final.ckpt.meta")

# Loads in accuracy Measurement:
theta = tf.get_default_graph().get_tensor_by_name('theta:0')

with tf.Session() as sess:
    saver.restore(sess, "saved_models/model_final.ckpt")
    X_new_data = data
    prediction = theta.eval(feed_dict={X: X_new_data})

output = pd.DataFrame(pd.np.column_stack([Eval_ObjectID, prediction]))
output.to_excel('Prediction.xlsx')

# np.allclose(best_theta, best_theta_restored)

#if __name__ == '__main__':
#    main()
