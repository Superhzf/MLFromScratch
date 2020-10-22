import numpy as np
from ..deep_learning.activation_functions import Sigmoid
from ..supervised_learning import XGBoostRegressionTree
from ..utils import to_categorical

# Q: Why is there a sigmoid func before p?
# A: Because we use a regression tree to do classification jobs and we need
# a probability result

# Q: what is the difference between lightgbm and xgboost?
# A: Gradient-based one-side sampling (GOSS) and Exclusive feature bundling (EFB)
# GOSS excludes a significant proportion of data instances with small gradients,
# and only use the rest to estimate the gain.
# EFB bundles mutually exclusive features (i.e. they rarely take  nonzero values
# simutaneously) to reduce the number of features. For example, if I have 3 variables,
# v1, v2, and v3, they cannot be 0 at the same time, after combination we have
# only one variable v4, if v4 is not zero, it means only one of v1, v2, and v3
# is not zero 
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
    def __init__(self,n_estimators=200,learning_rate=0.01,min_samples_split=2,
                 min_impurity=1e-7,max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split=min_samples_split
        self.min_impurity = min_impurity
        self.max_depth=max_depth

        self.loss = LogisticLoss()
        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=self.min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)

            self.trees.append(tree)

    def fit(self,X,y):
        y = to_categorical(y)

        y_pred = np.zeros(np.shape(y))
        for i in range(self.n_estimators):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y,y_pred),axis=1)
            tree.fit(X,y_and_pred)
            update_pred = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate,update_pred)


    def predict(self,X):
        y_pred = None
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_predict = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_predict)
            y_pred -= np.multiply(self.learning_rate, update_predict)

        # Turn into probability distribution (softmax)
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
