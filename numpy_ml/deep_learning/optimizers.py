import numpy as np

# Stochastic Gradient Descent with momentum
# The reason to use momentum is that if we are
# on the right direction, then the step would be larger and larger
# if the direction is wrong, momentum will keep us moving too far on the wrong
# direction
class StochasticGradientDescent():
    def __init__(self,learning_rate = 0.01,momentum=0.9):
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
# Answer: This kind of solves the problem of adagrad because with the help
# of moving average, the learning rates do not decrease so rapidly.
class RMSprop():
    def __init__(self,learning_rate = 0.01,rho = 0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def update(self,w,grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(w))

        self.Eg = self.rho*self.Eg + (1-self.rho)*np.power(grad_wrt_w,2)
        return w - self.learning_rate*grad_wrt_w/np.sqrt(self.Eg+self.eps)


# why adagrad?
# Answer: There are two reasons to use adagrad, the first one is that as the training
# goes on, we want to reduce the learning rate for the model to converge. The second
# reason is that frequent features have a smaller update and infrequent features
# have a larger update. It is usefull for sparse features.
# The downside of adagrad is that the learning rate becomes smaller and smaller
# and finally could stop, this won't work for saddle points
# TODO: why square the gradient and then take the square root of it instead of
# just calculte the abslute value of it?
class Adagrad():
    def __init__(self,learning_rate=0.01):
        self.learning_rate = learning_rate
        self.G = None # sum of squares of the gradients
        self.eps = 1e-8

    def update(self,w,grad_wrt_w):
        # If not initialized
        if self.G is None:
            self.G = np.zeros(np.shape(w))
        # Add the square of the gradient of the loss function at w
        self.G += np.power(grad_wrt_w,2)
        # Adaptive gradient with higher learning_rate for sparse data
        # If the feature is missing, then self.G becomes relatively smaller
        # and the step is relatively larger
        return w-self.learning_rate*grad_wrt_w/np.sqrt(self.G+self.eps)
