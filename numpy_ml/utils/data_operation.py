import numpy as np

def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """
    return np.var(X,axis=0)
