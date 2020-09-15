import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import log_loss
from .helpers import random_one_hot_matrix, random_stochastic_matrix
from numpy_ml.deep_learning.loss_functions import BinaryCrossEntropy
from numpy.testing import assert_almost_equal
import torch.nn as nn
import torch
from .conftest import test_log


def test_cross_entropy(cases):
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
