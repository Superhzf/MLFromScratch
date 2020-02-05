from __future__ import division
import numpy as np

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
