import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data in:
def start():
    """This takes user input for the file name so this program can be used on any dataset that is given.
    Usage example: 'datasets/prepared_data/Oklahoma_Working.xls'"""
    df = pd.read_excel(input("Please enter the path location of the file to be trained:"))
    return df

def preprocess(df):
    """This method takes the target dataframe and preprocesses it into vectors for the Algorithm to handle.
    """
    df = df[['PRONAME', 'ADDRESS', 'RESNAME']]
    
    def ngrams(string, n=3):
        string = re.sub(r',-./&',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def vectorize(x):
        """ This takes a dataframe that is of strings only and converts them into vectors. """
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
        return vectorizer.fit_trainsform(x)

    return df.apply(vectorize)

def fetch_batch(epoch, batch_index, batch_size):
    """This is for mini-batch gradient descent. 
    Usage- This takes the number of epochs defined later, the batch index and the size
    and randomly selects a subset of the dataset. It then cuts the indices and returns it to the model."""
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = X_train[indices]
    Y_batch = y_train.reshape(-1, 1)[indices]
    return X_batch, y_batch

df = pd.read_excel('datasets/prepared_data/Oklahoma_Working.xls') # Delete this later just right now it is just to save time.
 
# Train_set, Testing_Set split:
X = df[['ObjectID', 'PROPNAME', 'RESNAME', 'ADDRESS', 'Lat', 'Long']]
y = df[['duplicate_check']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle Object ID,
Eval_ObjectID = test_X['ObjectID']

# Drop all other information from Label data:
y_train = y_train.drop(columns=['PROPNAME', 'RESNAME', 'ADDRESS', 'OBJECTID', 'Lat', 'Long'])
y_test = y_test.drop(columns=['PROPNAME', 'RESNAME', 'ADDRESS', 'OBJECTID', 'Lat', 'Long'])

# Convert to Vectors
preproccess(X_train)
preprocess(X_test)

# Construction Phase for TF
feature_count = X_train.shape[0]
label_count = y_train.shape[0]

training_epochs = 10 # This will absoutely be played with during testing.
learning_rate = 0.01 # This value is set to low inorder to make sure the algorithm decends the gradient.
hidden_layers = feature_count -1
cost_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32, shape=(None, feature_count), name="X")
Y = tf.placeholder(tf.float32, shape=(None, label_count), name="Y")
theta = tf.Varaible(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.trainGradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
# save = tf.train.Saver() Turn this on when ready!

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

# Run TF
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epcoh, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})


    best_theta = theta.eval()

print(best_theta)
