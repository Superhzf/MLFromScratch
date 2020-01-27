import numpy as np

# Stochastic Gradient Descent with momentum
class StochasticGradientDescent():
    def __init__(self,learning_rate = 0.01,momentum=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.w_update = None

    def update(self,w,grad_wrt_w):
        # if not initialized
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))

        # Use momentum if set
        self.w_update = self.momentum * self.w_update + (1-self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate*self.w_update

# why RMSprop?
# Answer: 
class RMSprop():
    def __init__(self,learning_rate = 0.01,rho = 0.9):
        self.learning_rate = learning_rate
