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
