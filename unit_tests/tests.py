import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import log_loss
from .helpers import random_one_hot_matrix, random_stochastic_matrix,random_tensor
from numpy_ml.deep_learning.loss_functions import BinaryCrossEntropy
from numpy.testing import assert_almost_equal
import torch.nn as nn
import torch
from numpy_ml.deep_learning.activation_functions import Sigmoid, Softmax, ReLU, LeakyReLU, TanH
from numpy_ml.deep_learning.layers import Dense, Embedding, BatchNormalization
from numpy_ml.deep_learning.optimizers import StochasticGradientDescent


def test_binary_cross_entropy(cases):
    np.random.seed(12346)

    cases = int(cases)

    mine = BinaryCrossEntropy()
    gold = nn.BCELoss(reduction='sum')

    # ensure we get 0 when the two arrays are equal
    n_classes = 2
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_one_hot_matrix(n_examples, n_classes)
    y_tensor = torch.tensor(y)

    y_pred_tensor = torch.tensor(y_pred)
    assert_almost_equal(mine.loss(y, y_pred), gold(y_tensor, y_pred_tensor))

    # test on random inputs
    i = 1

    while i < cases:
        n_classes = 2
        n_examples = np.random.randint(1, 10)

        y = random_one_hot_matrix(n_examples, n_classes)
        y_tensor = torch.tensor(y)

        y_pred = random_stochastic_matrix(n_examples, n_classes)
        y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
        y_pred_tensor.retain_grad()

        gold_value = gold(y_pred_tensor,y_tensor)
        gold_value.backward()
        # compare forward value
        assert_almost_equal(np.sum(mine.loss(y, y_pred)), gold_value.detach().numpy())
        # compare backward value
        assert_almost_equal(mine.gradient(y, y_pred), y_pred_tensor.grad)

        i += 1
    print (' Successfully testing binary cross entropy function!')

def test_sigmoid_activation(cases):

    N = int(cases)

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        z_tensor = torch.tensor(z,requires_grad=True)
        z_tensor.retain_grad()

        mine = Sigmoid()
        gold = nn.Sigmoid()

        gold_value = gold(z_tensor)
        gold_value.retain_grad()

        loss_tensor = torch.square(gold_value).sum()/2.
        loss_tensor.backward()

        gold_grad = z_tensor.grad
        mine_value = mine(z)

        # compare forward
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        # compare backward
        assert_almost_equal(mine.gradient(z)*gold_value.grad.numpy(), gold_grad)
        i += 1
    print ('Successfully testing Sigmoid function!')

def test_softmax_activation(cases):
    N = int(cases)

    np.random.seed(12345)

    mine = Softmax()
    gold = nn.Softmax(dim=1)

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        z_tensor = torch.tensor(z,requires_grad=True)

        # let sum function as the loss function
        loss_tensor = torch.sum(z_tensor)
        loss_tensor.backward()
        gold_value = gold(z_tensor)
        mine_value = mine(z)
        # compare forward value
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        i += 1

    print ('Successfully testing Softmax function (forward only)!')

def test_relu_activation(cases):

    np.random.seed(12345)

    N = int(cases)
    i = 0
    while i < N:

        n_dims = np.random.randint(1, 100)
        z = np.random.randn(1, n_dims)
        z_tensor = torch.tensor(z,requires_grad=True)
        z_tensor.retain_grad()

        mine = ReLU()
        gold = nn.ReLU()

        gold_value = gold(z_tensor)
        gold_value.retain_grad()

        # Let the 1/2 times the sum of squares as the loss function
        loss_tensor = torch.square(gold_value).sum()/2.
        loss_tensor.backward()

        gold_grad = z_tensor.grad
        mine_value = mine(z)
        # compare forward value
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        # compare backward value
        assert_almost_equal(mine.gradient(z)*gold_value.grad.numpy(), gold_grad)
        i += 1
    print ('Successfully testing ReLU function!')

def test_leakyrelu_activation(cases):

    N = int(cases)

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = np.random.randn(1, n_dims)
        z_tensor = torch.tensor(z,requires_grad=True)
        z_tensor.retain_grad()

        alpha = np.random.uniform(0, 10)

        mine = LeakyReLU(alpha=alpha)
        gold = nn.LeakyReLU(negative_slope=alpha)

        gold_value = gold(z_tensor)
        gold_value.retain_grad()
        loss_tensor = torch.square(gold_value).sum()/2.
        loss_tensor.backward()
        gold_grad = z_tensor.grad
        mine_value = mine(z)

        # compare forward value
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        # compare backward value
        assert_almost_equal(mine.gradient(z), gold_grad/gold_value.grad)
        i += 1

    print ('Successfully testing LeakyReLU function!')

def test_tanh_activation(cases):

    N = int(cases)

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        z_tensor = torch.tensor(z,requires_grad=True)
        z_tensor.retain_grad()

        mine = TanH()
        gold = nn.Tanh()

        gold_value = gold(z_tensor)
        gold_value.retain_grad()

        loss_tensor = torch.square(gold_value).sum()/2.
        loss_tensor.backward()

        gold_grad = z_tensor.grad
        mine_value = mine(z)

        # compare forward
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        # compare backward
        assert_almost_equal(mine.gradient(z)*gold_value.grad.numpy(), gold_grad)
        i += 1
    print ('Successfully testing TanH function!')

def test_FullyConnected(cases):

    np.random.seed(12345)

    N = int(cases)

    decimal = 5

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        gold = nn.Linear(in_features=n_in, out_features=n_out, bias=True)
        mine = Dense(n_units = n_out, input_shape=(n_in,))
        # Do not allow the weights to be updated
        mine.trainable=False
        mine.initialize(None)

        # Adjust parameters to make them share the same set of weights and bias
        mine.W = gold.weight.detach().numpy().transpose()
        mine.b = gold.bias.detach().numpy()[None,:]

        # forward prop
        gold_value = gold(X_tensor)
        mine_value = mine.forward_pass(X)

        # loss
        gold_loss = torch.square(gold_value).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdw = gold.weight.grad.detach().numpy()
        gold_dLdb = gold.bias.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        dLdX = mine.backward_pass(mine_value)
        dLdW = mine.dw
        dLdb = mine.db


        # compare forward
        assert_almost_equal(mine_value,gold_value.detach().numpy(),decimal=decimal)
        # compare backward
        assert_almost_equal(dLdW.transpose(), gold_dLdw,decimal=decimal)
        assert_almost_equal(dLdb, gold_dLdb[None,:],decimal=decimal)
        assert_almost_equal(dLdX, gold_dLdX,decimal=decimal)
        i += 1

    print ('Successfully testing fully connected layer!')

def test_Embedding(cases):

    np.random.seed(12345)

    N = int(cases)
    decimal=5
    i = 1
    while i < N + 1:
        vocab_size = np.random.randint(1, 2000)
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        emb_dim = np.random.randint(1, 100)

        X = np.random.randint(0, vocab_size, (n_ex, n_in))
        X_tensor = torch.LongTensor(X)


        # initialize Embedding layer
        mine = Embedding(n_out=emb_dim, vocab_size=vocab_size)
        mine.initialize(None)
        mine.trainable = False
        gold = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)

        # Adjust parameters to make them share the same set of weights
        mine.W = gold.weight.detach().numpy()

        # forward prop
        mine_value = mine.forward_pass(X)
        gold_value = gold(X_tensor)

        # loss
        gold_loss = torch.square(gold_value).sum()/2.
        gold_loss.backward()
        # backward prop
        gold_dLdW = gold.weight.grad
        mine.backward_pass(mine_value)
        dLdW = mine.grad_W
        # For embedding, normally we don't have to calculate dLdX
        assert_almost_equal(mine_value, gold_value.detach().numpy(), decimal=decimal)
        i += 1
    print ('Successfully testing embedding layer!')

def test_BatchNorm(cases):

    np.random.seed(12345)

    N = int(cases)

    np.random.seed(12345)
    decimal=4

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(2, 1000)
        n_in = np.random.randint(1, 1000)

        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize BatchNorm layer
        gold = nn.BatchNorm1d(num_features=n_in, momentum=0.6)
        mine = BatchNormalization(momentum=1-0.6, input_shape=np.array([n_in,]))
        mine.trainable=True
        mine.initialize(None)

        # forward prop
        gold_value = gold(X_tensor)
        mine_value = mine.forward_pass(X)

        # loss
        gold_loss = torch.square(gold_value).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdgamma=gold.weight.grad
        gold_dLdbeta = gold.bias.grad
        gold_dLdX = X_tensor.grad.detach().numpy()

        dLdX = mine.backward_pass(mine_value)
        dLdgama = mine.grad_gamma
        dLdbeta = mine.beta

        # compare forward
        assert_almost_equal(mine_value, gold_value.detach().numpy(), decimal=decimal)
        # compare backward
        assert_almost_equal(dLdX, gold_dLdX, decimal=decimal)
        assert_almost_equal(dLdgama, gold_dLdgamma, decimal=decimal)
        assert_almost_equal(dLdbeta, gold_dLdbeta, decimal=decimal)
        i += 1

    print ('Successfully testing bacth normalization layer!')

def test_SGD_momentum(cases):
    """
    The idea is to do one epoch training and the compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    LR = 0.05
    MOMENTUM = 0.9

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        epochs = 2
        nesterov = np.random.choice(np.array([True,False]))
        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

        mine = Dense(n_units = n_out, input_shape=(n_in,))

        # initialize the SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    nesterov=nesterov)
        mine_optim = StochasticGradientDescent(learning_rate=LR,momentum=MOMENTUM,nesterov=nesterov)

        mine.trainable=True
        mine.initialize(mine_optim)


        # Adjust parameters to make them share the same set of weights and bias
        # we have to make a copy of the value, otherwise it will be changed to the updated one
        mine.W = model.weight.detach().numpy().transpose().copy()
        mine.b = model.bias.detach().numpy()[None,:].copy()
        gold_initial_weight = model.weight.detach().numpy().copy()

        # make sure initial weights are the same
        assert_almost_equal(mine.W, model.weight.detach().numpy().transpose(),decimal=decimal)
        assert_almost_equal(mine.b, model.bias.detach().numpy()[None,:],decimal=decimal)

        for this_epoch in range(epochs):
            optimizer.zero_grad()
            # forward prop
            model_value = model(X_tensor)
            mine_value = mine.forward_pass(X)
            # loss
            # a fake loss function, the target of it is just returning a single value and it should have gradients
            model_loss = torch.square(model_value).sum()/2.

            # backward prop
            model_loss.backward()
            optimizer.step()

            gold_weight = model.weight.detach().numpy()
            gold_bias = model.bias.detach().numpy()

            _ = mine.backward_pass(mine_value)

            mine_weight = mine.W
            mine_bias = mine.b

            # make comparison
            assert_almost_equal(mine_weight, gold_weight.transpose(),decimal=decimal)
            assert_almost_equal(mine_bias, gold_bias[None,:],decimal=decimal)
        i += 1
    print ('Successfully testing SGD optimizer!')
