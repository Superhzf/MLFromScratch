import copy
import numpy as np
import math

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
        # TODO: Kaiming initialization
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
        return X.dot(self.W)+self.b # Z = W*A + b

    def backward_pass(self,accum_grad):
        # accum_grad = dZ_curr
        # Save weights used during forward pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            dw = self.layer_input.T.dot(accum_grad)/self.input_shape
            db = np.sum(accum_grad, axis=0, keepdims=True)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, dw)
            self.b = self.b_opt.update(self.b, db)

        # Return accumulated gradient for the next layer
        # Calculation is based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T) # dA_prev
        return accum_grad

    def output_shape(self):
        return self.n_units

# The reason to use batchnormaliza is that without batchnormalization, the
# distribution of input to the next layer is constantly changing due to we
# are constantly updating the weights, so it would be hard and slow for the next layer
# to get optimized

# the reason to have beta and gamma is that the optimal mean and std is not
# necessarily 0 and 1. Besides, using beta and gamma make it easier to find
# the optimal value

# The impact is that we can use larger learning rate and small # of epochs
# Besides, it works as a regulzarization term because each hidden unit multiples
# a random value which is the sd and subtracts a value which is mean
# Also, the mean and var are learned across the training
class BatchNormalization(Layer):
    """Batch Normalization"""
    def __init__(self,momentum = 0.99):
        self.momentum = momentum
        self.trainable = True
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def initialize(self,optimizer):
        # Initialize the parameters
        self.gamma = np.ones(self.input_shape)
        self.betta = np.zeros(self.input_shape)
        # parameter optimizers
        self.gamma_opt = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)

    def forward_pass(self,X,training=True):
        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = np.mean(X,axis=0)
            self.running_var = np.var(X,axis=0)

        if training and self.trainable:
            mean = np.mean(X,axis=0)
            var = np.var(X,axis=0)
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*mean # it is based on batch
            self.running_var = self.momentum * self.running_var + (1-self.momentum)*var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean # this is batch mean, not running_mean for training
        self.stddev_inv = 1/np.sqrt(var+self.eps)

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

# reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def backward_pass(self,accum_grad):
        # save parameters used during the forward pass
        gamma = self.gamma

        # if the layer is trainable, updatet the parameters
        if self.trainable:
            X_norm = self.X_centered*self.stddev_inv
            grad_gamma = np.sum(accum_grad*X_norm,axis=0)
            grad_betta = np.sum(accum_grad,axis=0)
            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = accum_grad.shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights
        # stats from forward pass)
        accum_grad = (1 / batch_size) * gamma * self.stddev_inv * (
            batch_size * accum_grad
            - np.sum(accum_grad, axis=0)
            - self.X_centered * self.stddev_inv**2 * np.sum(accum_grad * self.X_centered, axis=0)
            )
        return accum_grad

    def out_shape(self):
        return self.input_shape

class Dropout(Layer):
    """
    A layer that randomly sets a fraction p of the output units of the previous
    layer to zero.

    Parameters:
    --------------------------
    p: float
        The probability that unit x is set to zero.
    """
    def __init__(self,p = 0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    # We only do drop out at the training stage and turn it off at the inference
    # stage because we want accuracy and results should be reproducible
    def forward_pass(self,X,training=True):

        if training:
            c = (1-self.p)
            self._mask = np.random.uniform(size = X.shape) > self.p
            c = self._mask/(1-p) # if p = 0.5, then the weights will be multiplied by 2
            X = X * c
        return X

    def backward_pass(self,accum_grad):
        return accum_grad*self._mask

    def output_shape(self):
        return self.input_shape

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    # 'selu': SELU,
    # 'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    # 'softplus': SoftPlus
}

class RNN(layer):
    """
    A vanilla fully-connected recurrent neural network layer.

    Parameters:
    --------------------------------------
    n_units: int
        The number of hidden states in a layer
    activation: string
        The name of the activation function which will be applied to the output
    of each state.
    bptt_trunc: int
        Decides how many time steps the gradient should be propagated backwards
    through states given the loss gradient for time step t
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit
    specifying the number of features of the input. Must be specified if it is
    the first layer in the network

    Reference:
    http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    """
    def __init__(self,n_units,activation='tanh',bptt_trunc=5,input_shape=None):
        self.input_shape=input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.W_p = None # Weight of the previous state
        self.W_o = None # Weight of the output
        self.W_i = None # Weight of the input

    def initialize(self,optimizer):
        _,input_dim = self.input_shape
        # Initialize the weights
        limit = 1/math.sqrt(input_dim)
        self.W_i = np.random.uniform(-limit,limit,(self.n_units,input_dim))
        limit = 1/math.sqrt(self.n_units)
        self.W_o = np.random.uniform(-limit,limit,(input_dim,self.n_units))
        self.W_p = np.random.uniform(-limit,limit,(self.n_units,self.n_units))
        # weight optimizers
        self.W_i_opt = copy.copy(optimizer)
        self.W_o_opt = copy.copy(optimizer)
        self.W_p_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape)+np.prod(self.U.shape)+np.prod(self.V.shape)

    def forward_pass(self,X,training=True):

        self.layer_input = X
        # By default, X is a group of batchs
        batch_size,timesteps,input_dim = self.layer_input.shape
        # cache values for use in backprop
        self.state_input = np.zeros((batch_size,timesteps,self.self.n_units))
        self.states = np.zeros(batch_size,timesteps+1,self.n_units)
        self.outputs = np.zeros(batch_size,timesteps,input_dim)

        # Set last timesteps to zero for calculation of the state_input at time
        # step zero
        self.states=[:,-1] = np.zeros((batch_size,self.n_units))

        for t in range(timesteps):
            # refL https://www.cs.toronto.edu/~tingwuwang/rnn_tutorial.pdf
            self.state_input[:,t] = X[:,t].dot(self.W_i)+self.states[:,t-1].dot(self.W.T)
            self.states[:,t] = self.activation(self.state_input[:,t])
            self.outputs[:,t] = self.states[:,t].dot(self.V.T)

        return self.outputs

    def backward_pass(self,accum_grad):
