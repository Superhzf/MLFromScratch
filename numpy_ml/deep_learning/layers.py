import copy
import numpy as np


class Layer(object):
    def set_input_shape(self,shape):
        """Sets the input shape of the layer"""
        self.input_shape = shape

    def layer_name(self):
        """The name of the layer. Used in model summary"""
        return self.__class__.name__

    def parameters(self):
        """The number of trainable parameters used by the layer"""
        return 0

    def backward_pass(self,accum_grad):
        """
        Propagates the accumulated gradient backward in the network.
        It returns the gradient with respect to the output of the previous layer
        """
        pass

    def output_shape(self):
        """The shape of the output produced by forward_pass"""
        pass


class Dense(Layer):
    """
    A fully-connected NN layer.
    Parameters:
    -------------------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        TODO
    """
    def __init__(self,n_units,input_shape=None):
        self.input_shape = input_shape
        self.layer_input = None
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.b = None

    def initialize(self,optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit,limit,(self.input_shape[0],self.n_units))
        self.b = np.zeros(1,self.n_units)
        # Weight optimizer
        self.W_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def parameters(self):
        # return the number of parameters in this layer
        return np.prod(self.W.shape)+np.prod(self.b.shape)

    def forward_pass(self,X,training=True):
        self.layer_input = X
        return X.dot(self.W)+self.b

    def backward_pass(self,accum_grad):
        # Save weights used during forward pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            dw = self.layer_input.T.dot(accum_grad)
            db = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, dw)
            self.b = self.b_opt.update(self.b, db)

        # Return accumulated gradient for the next layer
        # Calculation is based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return self.n_units
