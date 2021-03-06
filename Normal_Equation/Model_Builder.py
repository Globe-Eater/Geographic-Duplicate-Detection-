from datetime import datetime
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prep import start

tf.reset_default_graph()

def preprocess(df):
    """This method takes the target dataframe and preprocesses it into vectors for the Algorithm to handle.
     Will return a CRS sparse matrix for the matrix operations to be calulated on in the Tensorflow graph.
    """
    df = df[['PROPNAME', 'ADDRESS', 'RESNAME']]
    
    # Thank you to Alexander at 
    for col in df:
        df[col] = [np.nan if (not isinstance(val, str) and np.isnan(val)) else
          (val if isinstance(val, str) else str(int(val)))
          for val in df[col].tolist()]

    def ngrams(string, n=3):
        string = re.sub(r',-./&',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def vectorize(df):
        """ This takes a dataframe that is of strings only and converts them into vectors.
       The output of this funciton is a crs sparse matrix."""
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
        tf_idf_list = []
        for i in df:
            text = df[i]
            tf_idf_list.append(vectorizer.fit_transform(text))
        tf_idf_matrix = hstack([tf_idf_list[0], tf_idf_list[1]]).toarray()
        tf_idf_matrix = hstack([tf_idf_matrix, tf_idf_list[2]]).toarray()
        return tf_idf_matrix # 694

    def cosine(tf_idf_matrix):
        ''' X is the input for values in a dataframe for this function. The output is the cosine similarity between
        all X values in the matrix.'''
        similarity_matrix = []
        for i in tf_idf_matrix:
            for x in i:
            	similarity_matrix.append(cosine_similarity(x))
        return similarity_matrix

    tf_idf_matrix = vectorize(df)
    return tf_idf_matrix
    #similarity_matrix = cosine(tf_idf_matrix)
    #similarity_matrix = np.array(similarity_matrix)
    #return similarity_matrix

def main():
    ''' This method builds the model to detect duplicate records in the OLI data.'''
    
    # Tensorboard logs for viz and evaluation:
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    df = start()
    
    # Handle Object ID,
    Eval_ObjectID = df['OBJECTID']
    
    # Preprocess the data:
    data = preprocess(df)
    print('Preprocessing Complete.')

    # Shapes of Features (independent variables) and Labels (dependent variables)
    feature_count = data.shape[0]
    label_count = df['duplicate_check'].shape[0]
    
    # Here I am going to reverse the dimensions of X_train as a new variable:
    data_plus_bias = np.c_[np.ones((feature_count, 1)), data]
    data_plus_bias = np.transpose(data_plus_bias)
    
    # Train_set, Testing_Set split:
    X = data
    y = df[['duplicate_check']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_epochs = 1000 # This will absoutely be played with during testing.
    learning_rate = 0.01 # This value is set to low inorder to make sure the algorithm decends the gradient.
    
    X = tf.placeholder(tf.float32, shape=(None, label_count), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
    theta = tf.Variable(tf.random_uniform([label_count, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    
    with tf.name_scope("loss") as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")
    
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(mse)
    
    init = tf.global_variables_initializer()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver() # This will save the model
    
    batch_size = 100
    n_batches = int(np.ceil(feature_count / batch_size))
    
    def fetch_batch(epoch, batch_index, batch_size):
        """This is for mini-batch gradient descent. 
        Usage- This takes the number of epochs defined later, the batch index and the size
        and randomly selects a subset of the dataset. It then cuts the indices and returns it to the model."""
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(feature_count, size=batch_size)
        X_batch = data_plus_bias[indices]
        y_batch = df['duplicate_check'].values.reshape(-1, 1)[indices]
        return X_batch, y_batch

    print('Tensorflow session starting.')
    # Run TF
    with tf.Session() as sess:
        sess.run(init)
        '''if epoch % 100 == 0:   This goes in here somewhere.
                    print("Epoch", epoch, "MSE = ", mse.eval())
                    save_path = saver.save(sess, "tmp/my_model.ckpt")'''
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    
        best_theta = theta.eval()
        save_path = saver.save(sess, "saved_models/model_final.ckpt")
    
    file_writer.flush()
    file_writer.close()
    output = pd.DataFrame(pd.np.column_stack([Eval_ObjectID, best_theta]))
    output.to_excel('Evaluation.xlsx')

if __name__ == '__main__':
   main()
