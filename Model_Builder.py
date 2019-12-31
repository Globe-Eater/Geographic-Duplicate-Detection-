import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from prep import start

def preprocess(df):
    """This method takes the target dataframe and preprocesses it into vectors for the Algorithm to handle.
     Will return a CRS sparse matrix for the matrix operations to be calulated on in the Tensorflow graph.
    """
    df = df[['PROPNAME', 'ADDRESS', 'RESNAME']]

    def ngrams(string, n=3):
        string = re.sub(r',-./&',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def vectorize(df):
        """ This takes a dataframe that is of strings only and converts them into vectors.
       The output of this funciton is a crs sparse matrix."""
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
        tf_idf_matrix = []
        for i in df:
            text = df[i]
            tf_idf_matrix.append(vectorizer.fit_transform(text))
        return tf_idf_matrix

    def cosine(x):
        ''' X is the input for values in a dataframe for this function. The output is the cosine similarity between
        all X values in the matrix.'''
        return cosine_similarity(x)

    tf_idf_matrix = vectorize(df)
    return similarity_matrix = cosine(tf_idf_matrix)

def fetch_batch(epoch, batch_index, batch_size):
    """This is for mini-batch gradient descent. 
    Usage- This takes the number of epochs defined later, the batch index and the size
    and randomly selects a subset of the dataset. It then cuts the indices and returns it to the model."""
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(feature_count, size=batch_size)
    X_batch = X_train[indices]
    y_batch = y_train.values.reshape(-1, 1)[indices]
    return X_batch, y_batch

def main():
    ''' This method builds the model to detect duplicate records in the OLI data.'''
    df = start()

    # Train_set, Testing_Set split:
    X = df[['OBJECTID', 'PROPNAME', 'RESNAME', 'ADDRESS', 'Lat', 'Long']]
    y = df[['duplicate_check']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle Object ID,
    Eval_ObjectID = X_test['OBJECTID']

    # Convert to Vectors
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    # Construction Phase for TF
    feature_count = X_train.shape[0] # 3
    label_count = y_train.shape[0]   # 555

    n_epochs = 1000 # This will absoutely be played with during testing.
    learning_rate = 0.01 # This value is set to low inorder to make sure the algorithm decends the gradient.

    X = tf.placeholder(tf.float32, shape=(None, feature_count), name="X")
    y = tf.placeholder(tf.float32, shape=(None, label_count), name="Y")
    theta = tf.Variable(tf.random_uniform([feature_count, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    # save = tf.train.Saver() Turn this on when ready!

    batch_size = 50
    n_batches = int(np.ceil(label_count / batch_size))

    # Run TF
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    print(best_theta)

if __name__ == '__main__':
    main()
