import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import log_loss
from .helpers import random_one_hot_matrix, random_stochastic_matrix,random_tensor
from numpy_ml.deep_learning.loss_functions import BinaryCrossEntropy
from numpy.testing import assert_almost_equal
import torch.nn as nn
import torch
from numpy_ml.deep_learning.activation_functions import Sigmoid, Softmax, ReLU


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
    print ()
    print (' Successfully testing binary cross entropy function!')

def test_sigmoid_activation(cases):

    N = int(cases)

    mine = Sigmoid()
    gold = nn.Sigmoid()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_tensor((1, n_dims))
        z_tensor = torch.tensor(z,requires_grad=True)

        gold_output = gold(z_tensor)
        my_output = mine(z)
        # sum up the output value of sigmoid as the loss function
        loss = torch.sum(gold_output)
        loss.backward()
        # compare forward
        assert_almost_equal(my_output, gold_output.detach().numpy())
        # compare backward
        assert_almost_equal(mine.gradient(z), z_tensor.grad)
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
        # compare backward value
        assert_almost_equal(mine.gradient(z), z_tensor.grad)
        i += 1

    print ('Successfully testing Softmax function!')

def test_relu_activation(cases):

    np.random.seed(12345)

    N = int(cases)

    mine = ReLU()
    gold = nn.ReLU()

    i = 0
    while i < N:
        n_dims = np.random.randint(1, 100)
        z = random_stochastic_matrix(1, n_dims)
        z_tensor = torch.tensor(z,requires_grad=True)

        loss_tensor = torch.sum(z_tensor)
        loss_tensor.backward()
        gold_value = gold(z_tensor)
        mine_value = mine(z)
        # compare forward value
        assert_almost_equal(mine_value, gold_value.detach().numpy())
        # compare backward value
        assert_almost_equal(mine.gradient(z), z_tensor.grad)
        i += 1
    print ('Successfully testing ReLU function!')
