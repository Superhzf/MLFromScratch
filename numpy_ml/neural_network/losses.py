import numpy as np

def CrossEntropyLoss(y_hat,y):
    assert len(y_hat) == len(y)
    assert np.all(np.isin(np.unique(y),[0,1]))
    y_hat = np.array(y_hat)
    y = np.array(y)
    m = len(y)
    loss = 0
    mask_zero = y == 0
    loss += np.sum(np.log(1 - y_hat[mask_zero]))
    mask_one = y == 1
    loss += np.sum(np.log(y_hat[mask_one]))
    return -1*loss/m


def get_accuracy_value(y_hat,y,threshold = 0.5):
    y_hat = np.array(y_hat)
    y = np.array(y)
    class_hat = y_hat >= threshold
    return np.mean(class_hat == y)