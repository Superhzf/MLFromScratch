import copy
import numpy as np
import math
from .activation_functions import ReLU, Sigmoid, Softmax, TanH, LeakyReLU, FullSoftmax

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
        The number of neurons in the layer. It is also called n_out in other packages
    input_shape: (n_ex, n_in)
        The expected shape of the weight matrix. input_shape[1] is the the number of features.
    epoch: int
        This parameter indicates how many epochs have been finished
    """
    def __init__(self,n_units,input_shape=None, trainable=True):
        self.input_shape = input_shape
        self.layer_input = []
        self.n_units = n_units
        self.trainable = trainable
        self.W = None
        self.b = None

    def initialize(self,optimizer):
        # TODO: Kaiming initialization
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[1])
        self.W = np.random.uniform(-limit,limit,(self.input_shape[1],self.n_units))
        self.b = np.zeros((1,self.n_units))
        # Weight optimizer
        self.W_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        self.dw = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)


    def parameters(self):
        # return the number of parameters in this layer
        return np.prod(self.W.shape)+np.prod(self.b.shape)

    def forward_pass(self,X,training=True):
        self.layer_input.append(X)
        return X.dot(self.W)+self.b # Z = X*W + b

    def backward_pass(self,accum_grad):
        # accum_grad = dZ_curr
        # Save weights used during forward pass
        W = self.W.copy()
        this_layer_input = self.layer_input.pop()
        # Calculate gradient w.r.t layer weights
        self.dw += this_layer_input.T.dot(accum_grad)
        self.db += np.sum(accum_grad, axis=0, keepdims=True)
        if self.trainable:
            # Update the layer weights
            self.W = self.W_opt.update(self.W, self.dw)
            self.b = self.b_opt.update(self.b, self.db)

            self.dw = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)

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
    def __init__(self,momentum = 0.99, input_shape=None):
        self.momentum = momentum
        self.trainable = True
        self.eps = 1e-05
        self.running_mean = None
        self.running_var = None
        self.input_shape=input_shape

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
            # it is based on batch
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum)*mean
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

# reference:
# https://kratzert.github.io/2016/02/12/understanding-
# the-gradient-flow-through-the-batch-normalization-layer.html
    def backward_pass(self,accum_grad):
        # save parameters used during the forward pass
        gamma = self.gamma

        # if the layer is trainable, update the parameters
        if self.trainable:
            X_norm = self.X_centered*self.stddev_inv
            self.grad_gamma = np.sum(accum_grad*X_norm,axis=0)
            self.grad_beta = np.sum(accum_grad,axis=0)
            if self.gamma_opt is not None and self.beta_opt is not None:
                self.gamma = self.gamma_opt.update(self.gamma, self.grad_gamma)
                self.beta = self.beta_opt.update(self.beta, self.grad_beta)

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
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    # 'softplus': SoftPlus
}
good_act_fn_names = ['relu', 'sigmoid', 'softmax', 'tanh', 'leaky_relu']

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

class RNNCell(Layer):
    """
    This is the one to one RNNcell.
    Formula: https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html

    Parameters:
    --------------------------------------
    n_units: int
        The number of hidden states in a layer
    activation: string
        The name of the activation function which will be applied to the output
    bptt_trunc: int
        Decides how many time steps the gradient should be propagated backwards
    through states given the loss gradient for time step t
    input_shape: tuple
        n_ex * n_in. Must be specified if it is the first layer in the network
    """
    def __init__(self, n_units, activation='tanh', bptt_trunc=5, input_shape=None, trainable=True):
        self.input_shape=input_shape
        self.n_units=n_units
        self.activation=activation_functions[activation]()
        self.trainable = trainable
        self.bptt_trunc=bptt_trunc
        self.W_p=None
        self.W_i=None

    def initialize(self, optimizer):
        # initialize weights
        self.X=[]
        _, input_dim=self.input_shape
        limit = 1/math.sqrt(input_dim)
        self.W_i = np.random.uniform(-limit,limit,(input_dim, self.n_units))
        self.b_i = np.zeros((1,self.n_units))
        limit = 1/math.sqrt(self.n_units)
        self.W_p = np.random.uniform(-limit,limit,(self.n_units,self.n_units))
        self.b_p = np.zeros((1,self.n_units))
        # initialize optimizers
        self.W_i_opt = copy.copy(optimizer)
        self.b_i_opt = copy.copy(optimizer)
        self.W_p_opt = copy.copy(optimizer)
        self.b_p_opt = copy.copy(optimizer)
        # initialize gradients
        self.dW_i = np.zeros_like(self.W_i)
        self.db_i = np.zeros_like(self.b_i)
        self.dW_p = np.zeros_like(self.W_p)
        self.db_p = np.zeros_like(self.b_p)

        self.derived_variables={
            "A":[],
            "Z":[],
            "max_timesteps":0,
            "current_step":0,
            "dLdA_accumulator":[]
        }


    def parameters(self):
        return np.prod(self.W_i.shape)+np.prod(self.W_o.shape)+np.prod(self.W_p.shape)

    def forward_pass(self, Xt):
        """
        Xt: np.array of shape (n_ex, n_in), the input at timestemp t with n_ex observations
        and n_in features
        """
        self.derived_variables['max_timesteps']+=1
        self.derived_variables['current_step']+=1

        As=self.derived_variables["A"]
        if len(As) == 0:
            n_ex,  n_in=Xt.shape
            A0 = np.zeros((n_ex, self.n_units))
            As.append(A0)

        # We only calculate the value of hidden state
        # Under the condition that it is a many2many problem, refer to:
        # https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
        Zt = As[-1] @ self.W_p + self.b_p + Xt @ self.W_i + self.b_i
        At = self.activation(Zt)
        self.derived_variables["Z"].append(Zt)
        self.derived_variables["A"].append(At)

        self.X.append(Xt)
        return At

    def backward_pass(self, dLdAt):
        self.derived_variables["current_step"] -= 1
        Zs = self.derived_variables["Z"]
        As = self.derived_variables["A"]
        t = self.derived_variables["current_step"]
        # dLdA_accumulator is the gradient of loss w.r.t each hidden state
        dA_acc = self.derived_variables["dLdA_accumulator"]

        if len(dA_acc) == 0 :
            # dA_acc.append(np.zeros_like(As[0]))
            dA_acc.insert(0, dLdAt)

        # dA = dLdAt + dA_acc
        dA = dLdAt
        dZ = self.activation.gradient(Zs[t]) * dA
        assert dZ.size == Zs[t].size
        dXt = dZ @ self.W_i.T
        assert dXt.shape == self.X[t].shape

        self.dW_i=self.dW_i+self.X[t].T @ dZ
        self.dW_p=self.dW_p+As[t].T @ dZ
        self.db_i=self.db_i + np.sum(dZ, axis=0, keepdims=True)
        self.db_p=self.db_p + np.sum(dZ, axis=0, keepdims=True)
        dLdHidden = dZ@self.W_p.transpose()
        self.derived_variables["dLdA_accumulator"].insert(0, dLdHidden)
        return dXt, dLdHidden

    def update(self):
        if self.trainable:
            self.W_i = self.W_i_opt.update(self.W_i,self.dW_i)
            self.b_i = self.b_i_opt.update(self.b_i,self.db_i)
            self.W_p = self.W_p_opt.update(self.W_p,self.dW_p)
            self.b_p = self.b_p_opt.update(self.b_p,self.db_p)

    def output_shape(self):
        assert 1==0, "This function has not been implemented"

class many2oneRNN(Layer):
    """
    This is the RNN many to one layer, the use case is sentiment classification

    Parameters:
    ----------------------
    n_units: int
        The number of hidden states in a layer. It might be called n_out in other packages
    activation: str
        The name of activation functions. Choices could be relu, sigmoid,
    softmax, leaky_relu or tanh. The default value is tanh
    bptt_trunc: int
        This parameter means how far are dated to for backpropagation through time
    input_shape: (int, int)
        The shape of the input, the first int is the number of observations and the second
    int is the number of features.
    trainable: bool
        If true, then parameters will be updated through bptt.
    """
    def __init__(self,n_units, activation='tanh', bptt_trunc=5, input_shape=None, trainable=True):
        self.input_shape=input_shape
        self.n_units = n_units
        self.activation = activation
        self.trainable = True
        self.bptt_trunc = bptt_trunc
        self.trainable=trainable

    def initialize(self,optimizer):
        self.cell = RNNCell(n_units=self.n_units,
                            activation=self.activation,
                            bptt_trunc=self.bptt_trunc,
                            input_shape=self.input_shape,
                            trainable=self.trainable)
        self.cell.initialize(optimizer)

    def forward_pass(self, X):
        """
        Parameter:
        ----------------
        X: of shape (n_ex, n_in, n_t)
            The input for the forward process of RNN. n_ex shows the number of
        examples, n_in shows the number of features and n_t shows the number of
        timesteps.

        Output:
        -----------------
        np.dstack(Y): of shape (n_t, n_ex, n_out:
            The hidden state (or output) at each timestep. Since it is the many to
        one problem, so probably the output of the last timestep (np.dstack(Y)[-1,:,:])
        is what you want.
        """
        Y = []
        n_ex, n_in, self.n_t = X.shape
        for t in range(self.n_t):
            yt=self.cell.forward_pass(X[:, :, t])
            Y.append(yt)
        # The output of each hidden layer
        # return Y
        return np.dstack(Y)

    def backward_pass(self, dLdA):
        dLdX = []
        for t in reversed(range(self.n_t)):
            # dLdXt = self.cell.backward_pass(dLdA[:, :, t])
            dLdXt,dLdHidden_t = self.cell.backward_pass(dLdA)
            dLdA = dLdHidden_t
            dLdX.insert(0, dLdXt)
        dLdX=np.dstack(dLdX)
        self.cell.update()
        return dLdX

    def output_shape(self):
        assert 1==0, "This function has not been implemented"


class Embedding(Layer):
    def __init__(self, n_out, vocab_size, trainable=True):
        """
        Parameters:
        ---------------------
        n_out: int
            The output dimension
        vocab_size: int
            The total number of categories in the categorical variable (The total
            number of words in the vocabulary). All integer indices are expected
            to range between 0 and vocab_size - 1
        """
        self.n_out = n_out
        self.vocab_size = vocab_size
        self.W = None
        self.trainable = trainable

    def initialize(self, optimizer):
        limit = 1
        self.W = np.random.uniform(-limit,limit, (self.vocab_size, self.n_out))
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
        features. X is supposed to be an integer array. It is recommended that
        one embedding layer is only involved into one categorical variable, in
        other words, n_in should be one.

        Returns:
        ------------------------------
        Y: numpy array of shape (n_ex, n_in, n_out) represents n_ex observations,
        n_in features and n_out dimensions.
        """
        self.X = X
        Y = self.W[X]
        # if self.n_in is not None:
        #     n_ex, n_in, n_out = Y.shape
        #     Y = Y.reshape((n_ex, n_in * n_out))
        # # If the self.n_in is None, then the shape of Y meets the input requirement
        # # of LSTM/RNN
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

        if self.trainable:
            self.W = self.W_opt.update(self.W, self.grad_W)
            self.grad_W = np.zeros_like(self.W)

    def output_shape(self):
        if self.n_in is not None:
            return (self.n_in * self.n_out, )
        else:
            # self.n_in is None means the number of input categorical variables are
            # varied. So users have to set up the input shape of following
            # layers manually
            return (None, )


class LSTMCell(Layer):
    def __init__(self,
                 n_units,
                 input_shape,
                 act_fn = "tanh",
                 gate_fn="sigmoid",
                 trainable=True):

        """
        A single step (one to one) of a long short-term memory (LSTM) RNN

        Notes:
        ----------------------
        - X[t]: the input matrix at timestep t
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
            Gu[t] = gate_fn(Wu@Z[t]+bu) (It is called input gate in PyTorch)
            Go[t] = gate_fn(Wo@Z[t]+bo)
            Cc[t] = act_fn(Wc@Z[t]+bc)
            C[t] = Gf[t]*C[t-1]+Gu[t]*Cc[t]
            A[t] = Go[t]*act_fn(C[t])
        where @ indicates dot/matrix product, and * indicates elementwise multiplication

        Parameters:
        ----------------------
        n_units: int
            The dimension of a single hidden state/output on a given timestep.
        input_shape: (int, int)
            n_features, n_units
        act_fn: str
            The activation function. Default is 'Tanh'
        gate_fn: str
            The gate function for computing the update, output and forget gates.
            Default is sigmoid.
        """
        self.n_units = n_units
        self.input_shape = input_shape
        self.act_fn = activation_functions[act_fn]()
        self.gate_fn = activation_functions[gate_fn]()
        self.trainable = trainable

    def initialize(self, optimizer):
        self.X = []
        limit = 1 / math.sqrt(self.input_shape[1])
        self.W_ih = np.random.uniform(-limit,limit, (self.input_shape[1],4*self.n_units))
        self.b_ih = np.zeros((1, 4*self.n_units))

        self.W_hh = np.random.uniform(-limit,limit, (self.n_units,4*self.n_units))
        self.b_hh = np.zeros((1, 4*self.n_units))

        self.dW_ih = np.zeros_like(self.W_ih)
        self.db_ih = np.zeros_like(self.b_ih)

        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_hh = np.zeros_like(self.b_hh)

        self.W_ih_opt = copy.copy(optimizer)
        self.b_ih_opt = copy.copy(optimizer)

        self.W_hh_opt = copy.copy(optimizer)
        self.b_bh_opt = copy.copy(optimizer)

        self.derived_variables = {
            "A": [],
            "C": [],
            "Cc": [],
            "Gc": [],
            "Gf": [],
            "Go": [],
            "Gu": [],
            "current_step": 0,
            "dLdA_prev": [],
            "dLdC_prev": [],
            "max_timesteps": 0,
        }


    def forward_pass(self,Xt):
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

        self.derived_variables['max_timesteps']+=1
        self.derived_variables['current_step']+=1

        if len(self.derived_variables['A']) == 0:
            n_ex,n_in = Xt.shape
            init = np.zeros((n_ex,self.n_units))
            self.derived_variables['A'].append(init)
            self.derived_variables['C'].append(init)

        A_prev = self.derived_variables["A"][-1] # the last A
        C_prev = self.derived_variables['C'][-1] # the last C

        gates = Xt @ self.W_ih + self.b_ih + A_prev @ self.W_hh + self.b_hh
        ut, ft, cellt, ot= np.array_split(gates, 4, axis=1)
        Gut = self.gate_fn(ut)
        Gft = self.gate_fn(ft)
        Cct = self.act_fn(cellt)
        Got = self.gate_fn(ot)

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
        self.derived_variables['current_step'] -= 1
        t = self.derived_variables['current_step']
        return  self._bwd(t, dLdAt)

    def _bwd(self, t, dLdAt):
        A_prev = self.derived_variables['A'][t]
        # At = self.derived_variables['A'][t+1]
        C_prev = self.derived_variables['C'][t]
        Ct = self.derived_variables['C'][t+1]
        Cct = self.derived_variables['Cc'][t]
        Gft = self.derived_variables['Gf'][t]
        Got = self.derived_variables['Go'][t]
        Gut = self.derived_variables['Gu'][t]

        Xt = self.X[t]

        dLdA_prev_list = self.derived_variables["dLdA_prev"]
        dLdC_prev_list = self.derived_variables["dLdC_prev"]

        # Initialize accumulators
        if len(dLdA_prev_list) == 0:
            dLdA_prev_list.insert(0, dLdAt)
        dA = dLdAt

        if len(dLdC_prev_list) == 0:
            dC = dA * Got * self.act_fn.gradient(Ct)
            self.dLdC_prev = dC
        else:
            dC = dA * Got * self.act_fn.gradient(Ct) + self.dLdC_prev

        dLdC_prev_list.insert(0, dC)

        gates = Xt @ self.W_ih + self.b_ih + A_prev @ self.W_hh + self.b_hh
        ut, ft, cellt, ot= np.array_split(gates, 4, axis=1)

        dcellt = dC * Gut * self.act_fn.gradient(cellt)
        dft = dC * C_prev * self.gate_fn.gradient(ft)
        dot = dA * self.act_fn(Ct) * self.gate_fn.gradient(ot)
        dut = dC * Cct * self.gate_fn.gradient(ut)

        dgates = np.concatenate((dut, dft, dcellt, dot), axis=1)
        assert dgates.shape == gates.shape
        dXt = dgates @ self.W_ih.transpose()

        self.dW_ih = self.dW_ih + Xt.T @ dgates
        self.dW_hh = self.dW_hh + A_prev.T @ dgates

        self.db_ih = self.db_ih + dgates.sum(axis=0, keepdims=True)
        self.db_hh = self.db_hh + dgates.sum(axis=0, keepdims=True)

        dLdA_prev = dgates @ self.W_hh.transpose()
        self.derived_variables['dLdA_prev'].insert(0, dLdA_prev)
        self.dLdC_prev = dC * Gft
        return dXt, dLdA_prev

    def update(self):
        if self.trainable:
            self.Wc_opt.update(self.Wc, self.dWc)
            self.Wf_opt.update(self.Wf, self.dWf)
            self.Wo_opt.update(self.Wo, self.dWo)
            self.Wu_opt.update(self.Wu, self.dWu)

            self.bc_opt.update(self.bc, self.dbc)
            self.bf_opt.update(self.bf, self.dbf)
            self.bo_opt.update(self.bo, self.dbo)
            self.bu_opt.update(self.bu, self.dbu)


class many2oneLSTM(Layer):
    def __init__(self,n_units, input_shape, act_fn='tanh', gate_fn='sigmoid', trainable=True):
        """
        A single long short-term memory (LSTM) layer

        Parameters
        --------------------
        n_units: int
            The dimension of a single hidden state / output on a given timestamp.
        input_shape: tuple
            n_in * n_units
        act_fn: str
            The activation function for computing A[t]. Defacult is Tanh
        gate_fn: str
            The gate function for computing the update, forget, and output gates.
            The default value is Sigmoid
        """
        self.n_units = n_units
        self.input_shape = input_shape
        # self.bptt = bptt
        if act_fn not in good_act_fn_names:
            raise Exception('The activation function name is not understood')
        if gate_fn not in good_act_fn_names:
            raise Exception('The gate function name is not understood')
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        self.trainable = trainable

    def initialize(self, optimizer):
        self.cell = LSTMCell(
            n_units=self.n_units,
            input_shape=self.input_shape,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            trainable=self.trainable)
        self.cell.initialize(optimizer)
        self.curr_backward_t = 0

    def forward_pass(self,X):
        """
        Run a forward pass across all timesteps in the input.

        Parameters
        ----------------
        X: numpy.array of shape (n_ex, n_t, n_in)
           Input consisting of n_ex examples each of n_in dimensions
           and extending for n_t timesteps

        Returns
        ----------------
        Y: numpy.array of shape (n_ex, n_t, n_out)
           The value of the hidden state for each of the n_ex examples
           across each of the n_t timesteps
        """

        n_ex, n_in, self.n_t = X.shape
        H = [] # value of hidden state
        C = [] # cell memory value
        for t in range(self.n_t):
            ht, ct = self.cell.forward_pass(X[:, :, t])
            H.append(ht)
            C.append(ct)
        # n_ex, n_t, n_units = Y.shape
        # Y = Y.reshape((n_t, n_ex * n_units))
        return np.dstack(H), np.dstack(C)

    def backward_pass(self,dLdAt):
        """
        Run a backward pass across all timesteps in the input

        Parameters
        --------------
        dLdA: numpy.array of shape (n_ex, n_out)
            The gradient of the loss w.r.t. the final layer output

        Returns
        --------------
        dLdX: numpy.array of shape (n_ex, n_in, n_t)
            The value of the hidden state for each of the n_ex examples across
            each of the n_t examples
        """
        dLdX = []
        # each time when calculating gradients, dLdA_prev and dLdC_prev have to
        # be reset to be empty
        self.cell.derived_variables['dLdA_prev'] = []
        self.cell.derived_variables['dLdC_prev'] = []
        for t in reversed(range(self.n_t-self.curr_backward_t)):
            dLdXt, dLdA_prev = self.cell.backward_pass(dLdAt)
            dLdAt = dLdA_prev
            dLdX.insert(0,dLdXt)
        self.cell.update()
        dLdX = np.dstack(dLdX)
        self.curr_backward_t += 1
        self.cell.derived_variables['max_timesteps'] -= 1
        self.cell.derived_variables['current_step'] = self.cell.derived_variables['max_timesteps']
        return dLdX

    def output_shape(self):
        return (self.n_units, )

class BidirectionalLSTM(Layer):
    def __init__(self,
                 n_units,
                 input_shape,
                 act_fn='tanh',
                 gate_fn='sigmoid',
                 trainable=True):
        """
        A single bidirectional long short-term memory (LSTM) many-to-one layer.
        Parameters
        ----------
        n_units : int
            The number of features
        act_fn : str
            The activation function for computing ``A[t]``.
        gate_fn : str
            The gate function for computing the update, forget, and output gates.
        trainable : bool
            Whether to update the weights
        """
        self.n_units = n_units
        self.input_shape = input_shape
        # self.bptt = bptt
        if act_fn not in good_act_fn_names:
            raise Exception('The activation function name is not understood')
        if gate_fn not in good_act_fn_names:
            raise Exception('The gate function name is not understood')
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        self.trainable = trainable

    def parameters(self):
        raise Exception("This function has not been implemented")

    def initialize(self, optimizer):
        self.cell_fwd = LSTMCell(
            n_units=self.n_units,
            input_shape=self.input_shape,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            trainable=self.trainable)
        self.cell_fwd.initialize(optimizer)

        self.cell_bwd = LSTMCell(
            n_units=self.n_units,
            input_shape=self.input_shape,
            act_fn=self.act_fn,
            gate_fn=self.gate_fn,
            trainable=self.trainable)
        self.cell_bwd.initialize(optimizer)

    def forward_pass(self, X):
        H_fwd, H_bwd, C_fwd, C_bwd = [], [], [], []
        n_ex, n_in, self.n_t = X.shape
        for t in range(self.n_t):
            # forward LSTM
            ht_fwd, ct_fwd = self.cell_fwd.forward_pass(X[:, :, t])
            H_fwd.append(ht_fwd)
            C_fwd.append(ct_fwd)
            # backward LSTM
            ht_bwd, ct_bwd = self.cell_bwd.forward_pass(X[:, :, self.n_t-t-1])
            H_bwd.insert(0, ht_bwd)
            C_bwd.insert(0, ct_bwd)

        return np.dstack(H_fwd), np.dstack(H_bwd), np.dstack(C_fwd), np.dstack(C_bwd)

    def backward_pass(self, dLdA_fwd, dLdA_bwd):
        dLdX_fwd, dLdX_bwd, dLdX = [], [], []

        # forward direction
        dLdAt = dLdA_fwd.copy()
        for t in reversed(range(self.n_t)):
            dLdXt_fwd, dLdA_prev_fwd = self.cell_fwd.backward_pass(dLdAt)
            dLdAt = dLdA_prev_fwd
            dLdX_fwd.insert(0,dLdXt_fwd)
        self.cell_fwd.update()

        # backward direction
        dLdAt = dLdA_bwd.copy()
        for t in range(self.n_t):
            dLdXt_bwd, dLdA_prev_bwd = self.cell_bwd.backward_pass(dLdAt)
            dLdAt = dLdA_prev_bwd
            dLdX_bwd.append(dLdXt_bwd)
        self.cell_bwd.update()

        return np.dstack(dLdX_fwd),np.dstack(dLdX_bwd)

    def output_shape(self):
        raise Exception("This function has not been implemented")


class DotProductAttention(Layer):
    def __init__(self, emb_dim, d_k=None, d_v=None, trainable=True,num_heads=1):
        """
        Parameters:

        emb_dim: int
            The number of embedding features of the input.
        d_k: int
            The number of features for the query vectors
        d_v: int
            The number of features for the value vectors
        trainable: bool
            Whether to update the weights in the backpropagation process
        num_heads: int
            Number of parallel attention heads. Note that emb_dim will be
            split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
        ref:
            Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf
            https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L862
            https://www.youtube.com/watch?v=KmAISyVvE1Y&t=1090s
        ---------------
        """
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = self.emb_dim // num_heads
        assert self.head_dim * num_heads == self.emb_dim, "emb_dim must be divisible by num_heads"
        if d_k is None:
            self.d_k = self.emb_dim
        else:
            self.d_k = d_k
        if d_v is None:
            self.d_v = self.emb_dim
        else:
            self.d_v = d_v
        self.trainable=trainable

    def initialize(self, optimizer):
        self.softmax=FullSoftmax()

        if self.emb_dim == self.d_k and self.emb_dim == self.d_v:
            limit = 1 / math.sqrt(self.emb_dim)
            self.in_weight = np.random.uniform(-limit,limit,(self.emb_dim, 3 * self.emb_dim))
            self.out_weight = np.random.uniform(-limit,limit,(self.emb_dim, self.emb_dim))
            self.qkv_same = True
            self.scale = 1/np.sqrt(self.head_dim)

            self.dLdout_weight = np.zeros_like(self.out_weight)
            self.dLdin_weight = np.zeros_like(self.in_weight)

            self.in_weight_opt = copy.deepcopy(optimizer)
            self.out_weight_opt = copy.deepcopy(optimizer)
        else:
            limit_Q = 1 / math.sqrt(self.d_k)
            self.Q = np.random.uniform(-limit_Q,limit_Q,(self.emb_dim, self.d_k))
            self.K = np.random.uniform(-limit_Q,limit_Q,(self.emb_dim, self.d_k))
            limit_V = 1 / math.sqrt(self.d_v)
            self.V = np.random.uniform(-limit_V,limit_V,(self.emb_dim, self.d_v))
            self.qkv_same = False
            self.scale = 1/np.sqrt(self.d_k)


    def forward_pass(self, Q, K, V):
        """
        Compute the attention-weighted output of a collection of keys, values,
        and queries. In the most abstract sense,
            - Q(Query): Query vectors ask questions
            - K(Key): Key vectors advertise their relevancy to questions
            - V(Value): value vectors give possible answers to questions
        In words, keys and queries are combined via dot-product to produce a
        score, which is then passed through a softmax to produce a weight on each
        value vector in Values. We multiply each value vector
        by its weight, and then take the elementwise sum of each weighted value
        vector to get the d_v * output for the current example.
        ref:
        https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
        http://jalammar.github.io/illustrated-transformer/
        https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

        Parameters:
        ---------------
        Q: numpy.array of shape (target_seq, n_ex, emb_dim)
            A set of n_ex query vectors packed into a single matrix. Each query
            has target_seq words. Each word has emb_dim features from the
            embedding matrix. Please be aware that if num_heads>1, then emb_dim
            should be divisible by num_heads.
        K: numpy.array of shape (source_seq, n_ex, emb_dim)
            A set of n_ex key vectors packed into a single matrix. Each key has
            source_seq words. Each word has emb_dim features from the embedding
            matrix.Please be aware that if num_heads>1, then emb_dim
            should be divisible by num_heads.
        V: numpy.array of shape (source_seq, n_ex, emb_dim)
            A set of n_ex value vectors packed into a single matrix. Each key
            has source_seq words. Each word has emb_dim features from the
            embedding matrix.Please be aware that if num_heads>1, then emb_dim
            should be divisible by num_heads.

        Returns:
        ----------------
        Outputs: numpy.array of shape (target_seq, n_ex, emb_dim)
            The attention-weighted output values
        weights: numpy.array of shape (n_ex, target_seq, source_seq)
            For an observation, it shows the relevance of the source
            words to the target words.
        """
        if self.qkv_same:
            assert Q.shape == K.shape and Q.shape == V.shape
            assert Q.shape[-1] == self.emb_dim
            tgt_len, bsz, embed_dim = Q.shape
            src_len, _, _ = V.shape
            self.weights = []
            # Generate q,k,v
            if Q is K and Q is V:
                self.X = Q
                qkv = self.X @ self.in_weight
                # target_seq/source_seq x batch_size x feature_size
                self.q = qkv[:, :, : self.emb_dim]*self.scale
                self.k = qkv[:, :, self.emb_dim: 2*self.emb_dim]
                self.v = qkv[:, :, 2*self.emb_dim:]
            # reshaope q,k,v for multi-head attention
            self.q = self.q.reshape((tgt_len,bsz*self.num_heads,self.head_dim))
            self.k = self.k.reshape((src_len,bsz*self.num_heads,self.head_dim))
            self.v = self.v.reshape((src_len,bsz*self.num_heads,self.head_dim))
            # swaped: batch_szie x target_seq/source_seq x feature_size
            self.q = np.swapaxes(self.q, 0, 1)
            self.k = np.swapaxes(self.k, 0, 1)
            self.v = np.swapaxes(self.v, 0, 1)
            # calculate score
            # swaped k:batch_szie x feature_size x source_seq
            self.k = np.swapaxes(self.k, 1, 2)
            # scores: batch_szie x target_seq x source_seq
            self.scores = self.q @ self.k
            # target_len = self.scores.shape[1]
            for this_target_len in range(tgt_len):
                this_weights = self.softmax(self.scores[:, this_target_len, :])
                self.weights.append(this_weights)
            # target_seq x batch_size x source_seq
            self.weights = np.stack(self.weights)
            # swaped: batch_size x target_seq x source_seq
            self.weights = np.swapaxes(self.weights, 0, 1)
            # batch_size x target_seq x feature_size
            self.outputs = self.weights @ self.v
            # self.weights_gd is saved for calculating gradients
            self.weights_gd = self.weights

            self.weights = self.weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            self.weights = self.weights.sum(axis=1)/self.num_heads
            self.outputs = np.swapaxes(self.outputs, 0, 1)
            # In case num_heads>1
            self.outputs = self.outputs.reshape((tgt_len,bsz,embed_dim))
            return self.outputs @ self.out_weight, self.weights, self.scores


    def backward_pass(self, dLdOutput):
        """
        Backpropagation from layer outputs to inputs.

        Parameters:
        ----------------
        dLdOutput: numpy.array of shape (target_seq, n_ex, emb_dim)
            The gradients of the loss w.r.t. the first layer outputs
        """
        if self.qkv_same:
            this_dLdout_weight = np.swapaxes(self.outputs, 1, 2) @ dLdOutput
            self.dLdout_weight += np.sum(this_dLdout_weight, axis=0)
            # dLdoutputs: target_seq x n_ex x emb_dim
            dLdoutputs = dLdOutput @ self.out_weight.transpose()
            # weights_gd: batch_size x target_seq x source_seq
            weights_gd = np.swapaxes(self.weights_gd,1,2)
            tgt_seq, bsz, emd_dim = dLdoutputs.shape
            dLdoutputs = dLdoutputs.reshape((tgt_seq, bsz*self.num_heads, self.head_dim))
            # dLdv: batch_szie x src_len x head_dim
            dLdv =  weights_gd @ np.swapaxes(dLdoutputs,0,1)
            self.dLdv = dLdv
            # self.v: n_ex x source_seq x emb_dim
            # swaped_v: n_ex x emb_dim x source_seq
            swaped_v = np.swapaxes(self.v, 1, 2)
            # dLdweights: n_ex x target_seq x source_seq
            dLdweights = np.swapaxes(dLdoutputs,0,1) @ swaped_v
            # dLdscore
            target_seq = self.scores.shape[1]
            dweightdscore = []
            for this_target_len in range(target_seq):
                this_dweightdscore = self.softmax.gradient(self.scores[:, this_target_len, :],
                                                           dLdweights[:,this_target_len,:])
                dweightdscore.append(this_dweightdscore)
            # target_seq x n_ex x source_seq
            dweightdscore = np.stack(dweightdscore)
            # swaped dweightdscore: n_ex x target_seq x source_seq
            dLdscores = np.swapaxes(dweightdscore, 0, 1)
            # dLdq: n_ex x target_seq x emb_dim
            dLdq = dLdscores@np.swapaxes(self.k, 1, 2)
            dLdq = dLdq * self.scale
            self.dLdq = dLdq

            # dLdk: n_ex x source_seq x emb_dim
            dLdk = np.swapaxes(dLdscores, 1, 2)@self.q
            self.dLdk = dLdk

            # reshape q,k,v
            dLdq = np.swapaxes(dLdq, 1, 2)
            dLdk = np.swapaxes(dLdk, 1, 2)
            dLdv = np.swapaxes(dLdv, 1, 2)
            bszTheads, head_dim, tgt_len  = dLdq.shape
            _, _, src_len= dLdk.shape
            bsz = int(bszTheads/self.num_heads)
            assert bsz == bszTheads//self.num_heads
            emb_dim = head_dim * self.num_heads
            dLdq = dLdq.reshape((bsz, emb_dim, tgt_len))
            dLdk = dLdk.reshape((bsz, emb_dim, src_len))
            dLdv = dLdv.reshape((bsz, emb_dim, src_len))
            dLdq = np.swapaxes(dLdq, 1, 2)
            dLdk = np.swapaxes(dLdk, 1, 2)
            dLdv = np.swapaxes(dLdv, 1, 2)
            # n_ex x target_seq/source_seq x 3emb_dim (3:q,k,v)
            dLdqkv = np.concatenate((dLdq, dLdk, dLdv), axis=2)
            # swaped: n_ex x target_seq x emb_dim
            X = np.swapaxes(self.X, 0, 1)
            # swaped: n_ex x emb_dim x target_seq
            X = np.swapaxes(X, 1, 2)
            # before sum: n_ex x emb_dim x 3emb_dim
            self.dLdin_weight += np.sum(X @ dLdqkv, axis=0)
            dLdX = dLdqkv @ self.in_weight.transpose()
            dLdX = np.swapaxes(dLdX, 0, 1)

            return dLdX
    def update(self):
        if self.trainable:
            self.in_weight = self.in_weight_opt.update(self.in_weight,
                                                       self.dLdin_weight)
            self.out_weight = self.out_weight_opt.update(self.out_weight,
                                                         self.dLdout_weight)
            self.dLdin_weight = np.zeros_like(self.in_weight)
            self.dLdout_weight = np.zeros_like(self.out_weight)
