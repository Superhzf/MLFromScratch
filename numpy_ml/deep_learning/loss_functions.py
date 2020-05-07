from __future__ import division
import numpy as np
from scipy.special import expit

class Loss(object):
    def loss(self,y_true,y_pred):
        pass

    def gradient(self,y_true,y_pred):
        pass

class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self,y_true,y_pred):
        return 0.5*np.power((y_true-y_pred),2)

    def gradient(self,y_true,y_pred):
        return -(y_true-y_pred)

# Why Cross Entropy loss instead of MSE for binary classification problems?
# Answer: It is OK to use MSE but CE loss is better, the reason is that CE
# penalizes much to incorrect predictions, image the true value is 0 but your
# prediction is 1.

# Why CE loss generally?
# Answer: Because minimizing cross entropy loss is equal to maximizing log likelihood
# Proof: if y_true = 1, suppose P(y|x) = y_pred, if y_true = 0, P(y|x) = 1-y_pred
# If we combine them, we have P(y|x) = (y_pred^y)*(1-y_pred)^(1-y)
# logP(y|x) = log(y_pred^y) + log(1-y_pred)^(1-y) = ylogy+(1-y)log(1-y)
class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self,y,p):
        assert y.shape[1] == 2, 'BinaryCrossEntropy can only be used for binary classification problems'
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y*np.log(p)+(1-y)*np.log(1-p))

    def gradient(self,y,p):
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y/p)+(1-y)/(1-p)

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y * np.log(p + eps),axis=1)
        return cross_entropy

    def gradient(self, y, p):
        eps = np.finfo(float).eps
        return -np.sum(y/(p+eps),axis=1)


class BinomialDeviance(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        return -2 * np.mean((y * p) - np.logaddexp(0, p))

    def negative_gradient(self, y, p):
        return y - expit(p.ravel())

    def update_terminal_region(self, X, y, residual, tree_model):
        idx_list = np.array([tree_model.apply(sample) for sample in X])
        stack = [tree_model.root]
        node_list = []
        while len(stack) > 0:
            curr = stack.pop()
            if curr.value is not None:
                node_list.append(curr)
            if curr.true_branch is not None:
                stack.append(curr.true_branch)
            if curr.false_branch is not None:
                stack.append(curr.false_branch)

        for this_node in node_list:
            this_y = y[idx_list == this_node.leaf_idx]
            this_residual = residual[idx_list == this_node.leaf_idx]
            assert len(this_y) == len(this_residual) and len(this_y) > 0
            # The fomula comes from Newton's method and Taloyr extension
            numerator = np.sum(this_residual)
            denominator = np.sum((this_y - this_residual) * (1 - this_y + this_residual))
            if abs(denominator) < 1e-150:
                this_node.value = 0.0
            else:
                this_node.value = numerator / denominator
