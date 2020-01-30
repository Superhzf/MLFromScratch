import numpy as np

def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """
    return np.var(X,axis=0)


def calculate_entropy(y):
    """
    Calculate the entropy of label array y
    """
    unique_labels = np.unique(y)
    entropy = 0
    y_len = len(y)
    for label in unique_labels:
        p = sum(y == label)/y_len
        entropy += -p*np.log2(p)

    return entropy
