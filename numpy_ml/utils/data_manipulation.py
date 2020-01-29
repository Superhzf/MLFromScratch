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
