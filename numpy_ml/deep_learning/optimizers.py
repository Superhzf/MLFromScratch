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

    # ref: https://cs231n.github.io/neural-networks-3/
    def update(self,w,grad_wrt_w):
        # if not initialized
        if self.w_update is None:
            self.w_update = np.zeros(np.shape(w))

        # Use momentum if set
        self.w_update = self.momentum * self.w_update - self.learning_rate * grad_wrt_w
        # Move against the gradient to minimize loss
        return w + self.w_update

# why RMSprop?
# Answer: This solves the problem of adagrad, the learning rate is not becoming
# smaller and smaller. The idea is the same, if the gradient is large, we want
# the learning rate to be small, if the gradient is small, we want the learning rate
# to be large, so we get the same magnitude no matter how big or small that particular
# gradient is. It is pretty much like normalization.
#
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

# TODO: why accumulately add the squre of gradient instead of just use gradient
# for each iteration
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

# Adam is the combination of SGD with momentum and RMSprop
# Basically, it controls update and learning rate at the same time
class Adam():
    def __init__(self,learning_rate=0.001,b1=0.9,b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def update(self,w,grad_wrt_w):
        # if not initialized
        if self.m is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1*self.m+(1-self.b1)*grad_wrt_w
        self.v = self.b2*self.v+(1-self.b2)*np.power(grad_wrt_w,2)
        m_hat = self.m/(1-self.b1)
        v_hat = self.v/(1-self.b2)

        self.w_updt = self.learning_rate*m_hat/(np.sqrt(v_hat)+self.eps)

        return w - self.w_updt
