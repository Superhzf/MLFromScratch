from __future__ import division
import numpy as np

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
# 
class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self,y,p):
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y*np.log(p)+(1-y)*np.log(1-p))

    def gradient(self,y,p):
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y/p)+(1-y)/(1-p)
