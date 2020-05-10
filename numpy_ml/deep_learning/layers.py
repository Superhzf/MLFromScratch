import copy
import numpy as np
import math
from .activation_functions import ReLU, Sigmoid, Softmax, TanH

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
        The expected shape of the weight matrix
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
        self.b = np.zeros((1,self.n_units))
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

        # Calculate gradient w.r.t layer weights
        self.dw = self.layer_input.T.dot(accum_grad)
        self.db = np.sum(accum_grad, axis=0, keepdims=True)
        if self.trainable:
            # Update the layer weights
            self.W = self.W_opt.update(self.W, self.dw)
            self.b = self.b_opt.update(self.b, self.db)

        # Return accumulated gradient for the next layer
        # Calculation is based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T) # dA_prev
        return accum_grad

    def output_shape(self):
        return (self.n_units, )

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
        self.beta = np.zeros(self.input_shape)
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

    def output_shape(self):
        return self.input_shape

# reference: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    def backward_pass(self,accum_grad):
        # save parameters used during the forward pass
        gamma = self.gamma

        # if the layer is trainable, updatet the parameters
        if self.trainable:
            X_norm = self.X_centered*self.stddev_inv
            grad_gamma = np.sum(accum_grad*X_norm,axis=0)
            grad_beta = np.sum(accum_grad,axis=0)
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

    # def out_shape(self):
    #     return self.input_shape

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
    # stage because we want accuracy and results to be reproducible
    def forward_pass(self,X,training=True):
        if training:
            c = (1-self.p)
            self._mask = np.random.uniform(size = X.shape) > self.p
            c = self._mask/(1-self.p) # if p = 0.5, then the weights will be multiplied by 2
            X = X * c
        return X

    def backward_pass(self,accum_grad):
        return accum_grad*self._mask

    def output_shape(self):
        return self.input_shape


class Activation(Layer):
    """
    A layer that applies an activation function.

    Parameters:
    --------------------
    name: string
        The name of the activation function
    """
    def __init__(self,name):
        if name not in good_act_fn_names:
            raise Exception('The activation name is not understood')
        self.activation_name = name
        self.activation_func = activation_functions[name]()

    def layer_name(self):
        return "Activation {}".format(self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    # 'selu': SELU,
    # 'elu': ELU,
    'softmax': Softmax,
    # 'leaky_relu': LeakyReLU,
    'tanh': TanH,
    # 'softplus': SoftPlus
}
good_act_fn_names = ['relu', 'sigmoid', 'softmax', 'tanh']

# Why tanh is popular in RNN?
# A: ReLU is not a good choice for RNN, it will lead to gradient explosion because
# RNN layers share weights. This problem could be avoided by careful weight
# initialization. Compared with sigmoid, tanh outputs both positive and negative
# values make state values more flexible. For LSTM, tanh can make memory cell
# value decrease.

# What is the difference between BP and BPTT?
# A: BPTT still uses the chain rule, but the difference is that BPTT went through
# observations reversely and at the same time gradients are added
# up together becuase RNN shares weights and they are time sensitive.

# Why truncated W_i and W_p/ what is the benefits of truncated BPTT?
# A: truncated BPTT significantly speeds up the training process of RNN. The problem
# truncated BPTT is local optima, because we do not see the whole set

# Why RNN is easy to have gradient vanishing and exploding?
# A: Because RNN shares weights, for example if the weight for the input is 1 and
# the initialize state is 1, all inputs are 0, then after 1000 times, the result
# is still 1, if the weight is 1.01, then after 1000 times, it becomes a huge
# number, if the weight is 0.99, then after 1000 times, it becomes 0. For regular
# DNN, things are different, weights are not shared, so as long as we carefully
# initialize weights, most of time,we should be fine.
# Solution: Gradient clipping could help, RNN initialized with identity matrix + ReLU
# LSTM, advanced optimization

class many2manyRNN(Layer):
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
        return np.prod(self.W_i.shape)+np.prod(self.W_o.shape)+np.prod(self.W_p.shape)

    def forward_pass(self,X,training=True):

        self.layer_input = X
        # By default, X is a group of batchs
        batch_size, timestamps, feature_size = self.layer_input.shape
        # cache values for use in backprop
        self.state_input = np.zeros((batch_size, timestamps, self.n_units))
        self.states = np.zeros((batch_size, timestamps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timestamps, feature_size))

        # Set last timestamps to zero for calculation of the state_input at time
        # step zero
        self.states[:, -1, :] = np.zeros((batch_size,self.n_units))

        for t in range(timestamps):
            # ref https://www.cs.toronto.edu/~tingwuwang/rnn_tutorial.pdf
            # All input share self.W_i and self.W_p and self.W_o
            self.state_input[:, t, :] = X[:,t, :].dot(self.W_i.T)+self.states[:,t-1, :].dot(self.W_p.T)
            self.states[:,t, :] = self.activation(self.state_input[:,t, :])
            # Here might need an activation for classification problems
            self.outputs[:,t, :] = self.states[:,t, :].dot(self.W_o.T)

        return self.outputs

    def backward_pass(self,accum_grad):
        _,timestamps,_ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_W_p = np.zeros_like(self.W_p)
        grad_W_i = np.zeros_like(self.W_i)
        grad_W_o = np.zeros_like(self.W_o)

        # The gradient w.r.t the layer input
        # will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation through time
        for t in reversed(range(timestamps)):
            grad_W_o += accum_grad[:,t].T.dot(self.states[:,t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:,t].dot(self.W_o)*self.activation.gradient(self.state_input[:,t])
            # Calculate gradient w.r.t layer input
            accum_grad_next[:,t] = grad_wrt_state.dot(self.W_i)
            # Update gradient w.r.t W_i and W_p by backprop.
            for t_ in reversed(np.arange(max(0,t-self.bptt_trunc),t+1)):
                grad_W_i += grad_wrt_state.T.dot(self.layer_input[:,t_])
                grad_W_p += grad_wrt_state.T.dot(self.states[:,t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state=grad_wrt_state.dot(self.W_p)*self.activation.gradient(self.state_input[:,t_-1])
        # update weights
        self.W_i = self.W_i_opt.update(self.W_i,grad_W_i)
        self.W_o = self.W_o_opt.update(self.W_o,grad_W_o)
        self.W_p = self.W_p_opt.update(self.W_p,grad_W_p)

        return accum_grad_next

    def output_shape(self):
        # TODO: this seems to be not correct
        return self.input_shape


class many2oneRNN(Layer):
    def __init__(self,n_units,activation='tanh', output_dim=None,input_shape=None):
        self.input_shape=input_shape
        self.n_units = n_units
        self.activation = activation_functions[activation]()
        self.trainable = True
        self.output_dim = output_dim
        self.W_p = None # Weight of the previous state
        self.W_o = None # Weight of the output
        self.W_i = None # Weight of the input

    def initialize(self,optimizer):
        _,feature_dim = self.input_shape
        # Initialize the weights
        limit = 1/math.sqrt(feature_dim)
        self.W_i = np.random.uniform(-limit,limit,(self.n_units,feature_dim))
        limit = 1/math.sqrt(self.n_units)
        self.W_o = np.random.uniform(-limit,limit,(self.output_dim,self.n_units))
        self.W_p = np.random.uniform(-limit,limit,(self.n_units,self.n_units))
        # weight optimizers
        self.W_i_opt = copy.copy(optimizer)
        self.W_o_opt = copy.copy(optimizer)
        self.W_p_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W_i.shape)+np.prod(self.W_o.shape)+np.prod(self.W_p.shape)

    def forward_pass(self,X,training=True):

        self.layer_input = X
        # By default, X is a group of batchs
        batch_size, timestamps, feature_size = self.layer_input.shape
        self.timestamps = timestamps
        # cache values for use in backprop
        self.state_input = np.zeros((batch_size, timestamps, self.n_units))
        self.states = np.zeros((batch_size, timestamps+1, self.n_units))
        self.outputs = np.zeros((batch_size, self.output_dim))

        # Set last timestamps to zero for calculation of the state_input at time
        # step zero
        self.states[:, -1, :] = np.zeros((batch_size,self.n_units))

        for t in range(timestamps):
            # ref https://www.cs.toronto.edu/~tingwuwang/rnn_tutorial.pdf
            # All input share self.W_i and self.W_p and self.W_o
            self.state_input[:, t, :] = X[:,t, :].dot(self.W_i.T)+self.states[:,t-1, :].dot(self.W_p.T)
            self.states[:,t, :] = self.activation(self.state_input[:,t, :])
            # Here might need an activation for classification problems

        self.outputs = self.states[:,t, :].dot(self.W_o.T)

        return self.outputs

    def backward_pass(self,accum_grad):
        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_W_p = np.zeros_like(self.W_p)
        grad_W_i = np.zeros_like(self.W_i)
        grad_W_o = np.zeros_like(self.W_o)

        # Back Propagation through time
        grad_W_o = accum_grad.T.dot(self.states[:,self.timestamps-1])
        grad_wrt_state = accum_grad.dot(self.W_o)*self.activation.gradient(self.state_input[:,self.timestamps-1])
        for t in reversed(range(self.timestamps)):
            # Update gradient w.r.t W_i and W_p by backprop.
            grad_W_i += grad_wrt_state.T.dot(self.layer_input[:,t])
            grad_W_p += grad_wrt_state.T.dot(self.states[:,t-1])
            # Calculate gradient w.r.t previous state
            grad_wrt_state=grad_wrt_state.dot(self.W_p)*self.activation.gradient(self.state_input[:,t-1])

        accum_grad_next = grad_wrt_state.dot(self.W_i)
        # update weights
        self.W_i = self.W_i_opt.update(self.W_i,grad_W_i)
        self.W_o = self.W_o_opt.update(self.W_o,grad_W_o)
        self.W_p = self.W_p_opt.update(self.W_p,grad_W_p)

        return accum_grad_next

    def output_shape(self):
        # TODO: This seems to be not correct
        return self.input_shape


class Embedding(Layer):
    def __init__(self, n_out, vocab_size, n_in):
        """
        Parameters:
        ---------------------
        n_out: int
            The output dimension
        vocab_size: int
            The total number of categories in the categorical variable (The total
            number of words in the vocabulary). All integer indices are expected
            to range between 0 and vocab_size - 1
        n_in: int
            The number of categorical variables
        """
        self.n_out = n_out
        self.vocab_size = vocab_size
        self.n_in = n_in
        self.W = None

    def initialize(self, optimizer):
        limit = 1
        self.W = np.random.uniform(-limit,limit,(self.vocab_size, self.n_out))
        self.grad_W = np.zeros_like(self.W)
        self.W_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape)

    def forward_pass(self, X, training=True):
        """
        Compute the layer output on a single minibatch. Y = W[X]

        Parameters:
        ------------------------------
        X: numpy array of shape (n_ex, n_in) represents n_ex examples and n_in
        features. X is supposed to be an integer array.

        Returns:
        ------------------------------
        Y: numpy array of shape (n_ex, n_in*n_out) represents n_ex observations,
        n_in features and n_out dimensions.
        """
        self.X = X
        Y = self.W[X]
        n_ex, n_in, n_out = Y.shape
        Y = Y.reshape((n_ex, n_in * n_out))
        return Y

    def backward_pass(self, accum_grad):
        """
        If used, it must be the first layer of the architecture, so it doesn't
        have to return gradients w.r.t. the input
        """
        for dy, x in zip(accum_grad, self.X):
            dW = np.zeros_like(self.grad_W)
            dy = dy.reshape(-1, self.n_out)
            for ix, v_id in enumerate(x.flatten()):
                dW[v_id] += dy[ix]
            self.grad_W += dW

        self.W = self.W_opt.update(self.W, self.grad_W)

    def output_shape(self):
        return (self.n_in * self.n_out, )


class LSTMCell(Layer):
    def __init__(self,
                 n_units,
                 input_shape,
                 act_fn = "tanh",
                 gate_fn="sigmoid"):

        """
        A single step of a long short-term memory (LSTM) RNN

        Notes:
        ----------------------
        - X[t]: the input matrix at timestep x
        - Z[t]: the input to each of the gates at timestep t
        - A[t]: the value of the hidden state at timestep t
        - Cc[t]: the value of the *candidate* cell/memory state at timestep t
        - C[t]: the value of the *final* cell/memory state at timestep t
        - Gf[t]: the output of the forget gate at the timestep t
        - Gu[t]: the output of the update gate at timestep t
        - Go[t]: the output of the output gate at timestep t

        Equations:
            Z[t] = hstack(A[t-1],X[t]) # stack arrays horizontally
            Gf[t] = gate_fn(Wf@Z[t]+bf)
            Gu[t] = gate_fn(Wu@Z[t]+bu)
            Go[t] = gate_fn(Wo@Z[t]+bo)
            Cc[t] = act_fn(Wc@Z[t]+bc)
            C[t] = Gf[t]*C[t-1]+Gu[t]*Cc[t]
            A[t] = Go[t]*act_fn(C[t])
        where @ indicates dot/matrix product, and * indicates elementwise multiplication

        Parameters:
        ----------------------
        n_out: int
            The dimension of a single hidden state/output on a given timestep.
        act_fn: str
            The activation function. Default is 'Tanh'
        gate_fn: str
            The gate function for computing the update, output and forget gates.
        Default is sigmoid.
        init: {'glorot_normal','glorot_uniform','he_normal','he_uniform'}
            The weight initialization stretegy
        optimizer: str
            The optimization strategyto use when performing gradient updates
        """
        self.n_units = n_units
        self.input_shape = input_shape
        self.act_fn = activation_functions[act_fn]()
        self.gate_fn = activation_functions[gate_fn]()
        self.trainable = True

    def initialize(self, optimizer):
        self.X = []
        limit = 1 / math.sqrt(self.input_shape[0])
        self.Wc = np.random.uniform(-limit,limit, (self.input_shape[0]+self.n_units, self.n_units))
        self.Wf = np.random.uniform(-limit,limit, (self.input_shape[0]+self.n_units, self.n_units))
        self.Wo = np.random.uniform(-limit,limit, (self.input_shape[0]+self.n_units, self.n_units))
        self.Wu = np.random.uniform(-limit,limit, (self.input_shape[0]+self.n_units, self.n_units))

        self.bc = np.zeros((1, self.n_units))
        self.bf = np.zeros((1, self.n_units))
        self.bo = np.zeros((1, self.n_units))
        self.bu = np.zeros((1, self.n_units))

        self.dWc = np.zeros_like(self.Wc)
        self.dWf = np.zeros_like(self.Wf)
        self.dWo = np.zeros_like(self.Wo)
        self.dWu = np.zeros_like(self.Wu)

        self.dbc = np.zeros_like(self.bc)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)
        self.dbu = np.zeros_like(self.bu)

        # weight optimizers
        self.Wc_opt = copy.copy(optimizer)
        self.Wf_opt = copy.copy(optimizer)
        self.Wo_opt = copy.copy(optimizer)
        self.Wu_opt = copy.copy(optimizer)

        self.bc_opt = copy.copy(optimizer)
        self.bf_opt = copy.copy(optimizer)
        self.bo_opt = copy.copy(optimizer)
        self.bu_opt = copy.copy(optimizer)

        self.derived_variables = {
            "A": [],
            "C": [],
            "Cc": [],
            "Gc": [],
            "Gf": [],
            "Go": [],
            "Gu": [],
            "current_step": 0,
            "dLdA_accumulator": None,
            "dLdC_accumulator": None,
            "n_timesteps": 0,
        }


    def forward_pass(self,Xt, training=True):
        """
        Compute the layer output for a single timestep

        Parameters
        ----------------------
        Xt: np.array of shape (n_ex,n_in). Input at timestap t consisting of n_ex
            observations each of dimensionality n_in

        Returns
        ------------------------
        At: np.array of shape (n_ex,n_out). The value of hidden state at timestep
            t for each of the n_ex observations
        Ct: np.array of shape (n_ex,n_out). The value of the cell/memory state at
            timestep t for each of the n_ex observations
        """

        self.derived_variables['n_timesteps']+=1
        self.derived_variables['current_step']+=1

        if len(self.derived_variables['A']) == 0:
            n_ex,n_in = Xt.shape
            init = np.zeros((n_ex,self.n_units))
            self.derived_variables['A'].append(init)
            self.derived_variables['C'].append(init)

        A_prev = self.derived_variables["A"][-1] # the last A
        C_prev = self.derived_variables['C'][-1] # the last C

        # concatenate A_prev and Xt to create Zt
        Zt = np.hstack([A_prev, Xt])
        Gft = self.gate_fn(Zt@self.Wf+self.bf)
        Gut = self.gate_fn(Zt@self.Wu+self.bu)
        Got = self.gate_fn(Zt@self.Wo+self.bo)
        Cct = self.act_fn(Zt@self.Wc+self.bc)
        Ct = Gft*C_prev+Gut*Cct
        At = Got*self.act_fn(Ct)

        # bookkeeping
        self.X.append(Xt)
        self.derived_variables['A'].append(At)
        self.derived_variables['C'].append(Ct)
        self.derived_variables['Cc'].append(Cct)
        self.derived_variables['Gf'].append(Gft)
        self.derived_variables['Gu'].append(Gut)
        self.derived_variables['Go'].append(Got)

        return At, Ct

    def backward_pass(self,dLdAt):
        """
        Run a backward pass across all timesteps in the input

        Parameters
        --------------------------------
        dLdAt:np.array of shape (n_ex,n_out). The gradient of the loss function
                w.r.t the layer output (i.e. hidden states) at timestep t

        Returns
        --------------------------------
        dLdXt:np.array of shape (n_ex,n_in). The gradient of the loss w.r.t. the
              layer input at timestep t
        """

        self.derived_variables['current_step']-=1
        t = self.derived_variables['current_step']

        A_prev = self.derived_variables['A'][t]
        At = self.derived_variables['A'][t+1]
        C_prev = self.derived_variables['C'][t]
        Ct = self.derived_variables['C'][t+1]
        Cct = self.derived_variables['Cc'][t]
        Gft = self.derived_variables['Gf'][t]
        Got = self.derived_variables['Go'][t]
        Gut = self.derived_variables['Gu'][t]

        Xt = self.X[t]
        Zt = np.hstack([A_prev,Xt])

        dA_acc = self.derived_variables["dLdA_accumulator"]
        dC_acc = self.derived_variables["dLdC_accumulator"]

        # Initialize accumulators
        if dA_acc is None:
            dA_acc = np.zeros_like(At)

        if dC_acc is None:
            dC_acc = np.zeros_like(Ct)

        # Gradient calculations
        # ---------------------------
        dA = dLdAt + dA_acc
        dC = dC_acc + dA * Got * self.act_fn.gradient(Ct)

        # Compute the input to the gate functions at timestamp t
        _Gc = Zt @ self.Wc + self.bc
        _Gf = Zt @ self.Wf + self.bf
        _Go = Zt @ self.Wo + self.bo
        _Gu = Zt @ self.Wu + self.bu

        # Compute gradients w.r.t. the input to each gate
        dCct = dC * Gut * self.act_fn.gradient(_Gc)
        dGft = dC * C_prev * self.gate_fn.gradient(_Gf)
        dGot = dA * self.act_fn(Ct) * self.gate_fn.gradient(_Go)
        dGut = dC * Cct * self.gate_fn.gradient(_Gu)


        dZ = dGft @ self.Wf.T + dGut @ self.Wu.T + dCct @ self.Wc.T + dGot @ self.Wo.T
        dXt = dZ[:, self.n_units:]

        self.dWc += Zt.T @ dCct
        self.dWf += Zt.T @ dGft
        self.dWo += Zt.T @ dGot
        self.dWu += Zt.T @ dGut
        self.dbc += dCct.sum(axis=0, keepdims = True)
        self.dbf += dGft.sum(axis=0, keepdims = True)
        self.dbo += dGot.sum(axis=0, keepdims = True)
        self.dbu += dGut.sum(axis=0, keepdims = True)

        if self.trainable:
            self.Wc_opt.update(self.Wc, self.dWc)
            self.Wf_opt.update(self.Wf, self.dWf)
            self.Wo_opt.update(self.Wo, self.dWo)
            self.Wu_opt.update(self.Wu, self.dWu)

            self.bc_opt.update(self.bc, self.dbc)
            self.bf_opt.update(self.bf, self.dbf)
            self.bo_opt.update(self.bo, self.dbo)
            self.bu_opt.update(self.bu, self.dbu)

        self.derived_variables['dLdA_accumulator'] = dZ[:,:self.n_units]
        self.derived_variables['dLdC_accumulator'] = Gft * dC

        return dXt

class LSTM(Layer):
    def __init__(self,n_units, input_shape, act_fn='tanh',gate_fn='sigmoid', optimizer=None):
        """
        A single long short-term memory (LSTM) RNN layer

        Parameters
        --------------------
        n_out: int
            The dimension of a single hidden state / output on a given timestamp.
        act_fn: str
            The activation function for computing A[t]. Defacult is Tanh
        gate_fn: str
            The gate function for computing the update, forget, and output gates.
            The default value is Sigmoid
        init: {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. The default value is glorot_uniform
        optimizer: str
            The optimization strategy to use when performing gradient updates.
        """
        super().__init__(optimizer)

        self.init = init
        self.n_in = n_in
        self.n_out = n_out
        self.n_timesteps = None
        if act_fn not in good_act_fn_names:
            raise Exception('The activation function name is not understood')
        if gate_fn not in good_act_fn_names:
            raise Exception('The gate function name is not understood')
        self.act_fn = act_fn
        self.gate_fn = gate_fn

    def initialize(self, optimizer):
        self.cell = LSTMCell(
            n_in=self.n_in,
            n_out=self.n_out,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn)
        self.cell.initialize(optimizer)

    def forward_pass(self,X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------------
        X: numpy.array of shape (n_ex, n_in, n_t)
           Input consisting of n_ex examples each of dimensionality n_in
           and extending for n_t timesteps

        Returns
        ----------------
        Y: numpy.array of shape (n_ex, n_out, n_t)
           The value of the hidden state for each of the n_ex examples
           across each of the n_t timesteps
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()

        Y = []
        n_ex, n_in, n_t = X.shape
        for t in range(n_t):
            yt, _ = self.cell.forward(x[:,:,t])
            Y.append(yt)
        return np.dstack(Y)

    def backward_pass(self,dLdA):
        """
        Run a backward pass across all timesteps in the input

        Parameters
        --------------
        dLdA: numpy.array of shape (n_ex, n_out, n_t)
            The gradient of the loss w.r.t. the layer output for each of the
            n_ex examples across all n_t timesteps

        Returns
        --------------
        dLdX: numpy.array of shape (n_ex, n_in, n_t)
            The value of the hidden state for each of the n_ex examples across
            each of the n_t examples
        """
        assert self.cell.trainable, "Layer is frozen"
        dldX = []
        n_ex, n_out, n_t = dLdA.shape
        for t in reversed(range(n_t)):
            dLdXt, _ = self.cell.backward(dLdA[:,:,t])
            dLdX.insert(0,dLdXt)
        dLdX = np.dstak(dLdX)
        return dLdX
