import numpy as np

class Sigmoid():
    def __call__(self,x):
        return 1/(1+np.exp(-x))

    def gradient(self,x):
        p = self.__call__(x)
        return p*(1-p)

# The reason to use softmax instead of standard normalization for multi-class
# classification is that 
class Softmax():
    def __call__(self,x):
        exp_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=-1,keepdims=True)

    def gradient(self,x):
        p = self.__call__(x)
        return p*(1-p)

class ReLU():
    def __call__(self,x):
        return np.maximum(x,0)

    def gradient(self,x):
        z = np.ones(x.shape)
        z[x<0] = 0
        return x
