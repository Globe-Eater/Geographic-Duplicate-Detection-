import tensorflow as tf
import pandas as pd
from prep import start
from Model_Builder import preprocess


#def main():
'''This program will predict duplicate records that are input into it.
Requires: prepared data from prep.py.
Output: an excel file of objectID and prediction.
'''

df = start()

#df = df[df.OBJECTID == 82852]

# Handle Object ID,
Eval_ObjectID = df['OBJECTID']

# Preprocess the data:
data = preprocess(df)

# Shapes of Features (independent variables) and Labels (dependent variables)
label_count = df['duplicate_check'].shape[0]

# Construct Variables This needs to be done for the restored Model.
X = tf.placeholder(tf.float32, shape=(None, label_count), name="X")

# To make the shapes aline
data = data.transpose()

# Reinitialize the new variables:
init = tf.global_variables_initializer()

# This loads the graph structure
saver = tf.train.import_meta_graph("saved_models/model_final.ckpt.meta")

# Loads in accuracy Measurement:
theta = tf.get_default_graph().get_tensor_by_name('theta:0')

with tf.Session() as sess:
    saver.restore(sess, "saved_models/model_final.ckpt")
    init.run()
    prediction = theta.eval(feed_dict={X: data})


# Going to slice the Eval_ObjectID to merge with the size of the trained predictions.
#size = prediction.shape
#size = size[0]
#Eval_ObjectID = Eval_ObjectID[:size] 

output = pd.DataFrame(pd.np.column_stack([Eval_ObjectID, prediction]))
output = output.rename(columns=({0: 'OBJECTID', 1: 'Prediction'}))
result = df.merge(output, on=['OBJECTID'], how='left')
result = df.sort_values(by=['duplicate_check'], ascending=False)
result.to_excel('Prediction.xlsx')

#if __name__ == '__main__':
#    main()
