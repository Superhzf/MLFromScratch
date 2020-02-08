from __future__ import division
import numpy as np
from iterations import combinations_with_replacement

def batch_generator(X,y=None,batch_size = 64):
    """Simple batch geneartor"""
    n_samples = X.shape[0]
    for i in np.arange(0,n_samples,batch_size):
        begin,end = i,min(i+batch_size,n_samples)
        if y is not None:
            yield X[begin:end],y[begin:end]
        else:
            yield X[begin:end]

def divide_on_feature(X,feature_i,threshold):
    """
    Divide dataset X based on if sample value on feature_i is larger than
    the given threshold
    """
    split_func = None
    if isinstance(threshold,int) or isinstance(threshold,float):
        X_1 = X[X[:,feature_i]>=threshold]
        X_2 = X[X[:,feature_i]<threshold]
    else:
        X_1 = X[X[:,feature_i]==threshold]
        X_2 = X[X[:,feature_i]!=threshold]

    return np.array([X_1,X_2])


def to_categorical(x,n_col=None):
    """
    one-hot encoding of norminal values
    """
    if not n_col:
        n_col = np.amax(x)+1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def polynomial_features(X,degree):
    n_samples,n_features = X.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features),i) for i in range(0,degree+1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinnations = index_combinations()
    n_output_features = len(combinnations)
    X_new = np.empty((n_samples,n_output_features))

    for i,index_combs in enumerate(combinations):
        X_new[:,i] = np.prod(X[:,index_combs],axis=1)

    return X_new


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    # l2[l2 == 0] = 1???
    return X / np.expand_dims(l2, axis)
