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
        return -(y-p)

    # w.r.t y_pred
    def hess(self,y,y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)
