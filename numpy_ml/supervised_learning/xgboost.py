import numpy as np
from deep_learning.activation_functions import Sigmoid

class LogisticLoss():
    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient

    def loss(self,y,y_pred):
        y_pred = np.clip(y_pred,1e-15,1-1e-15)
        p = self.log_func(y_pred)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    # gradient w.r.t y_pred
    def gradient(self,y,y_pred):
        p = self.log_func(y_pred)
        return p-y

    # w.r.t y_pred
    def hess(self,y,y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)

class XGBoost(object):
    """
    XGBoost classifier
    reference: https://arxiv.org/abs/1603.02754

    Parameters:
    --------------------------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be takend when following the negative gradient
    during training
    min_sample_split: int
        The minimum number of samples needed to make a split when building a tree
    min_impurity: float
        The minimum impurity required to split the tree further
    max_depth: int
        The maximum depth of a tree
    """
    def __init__(self,n_estimators=200,learning_rate=0.01,min_sample_split=2,
                 min_impurity=1e-7,max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_sample_split=min_sample_split
        self.min_impurity = min_impurity
        self.max_depth=max_depth

        self.loss = LogisticLoss()
        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree =
