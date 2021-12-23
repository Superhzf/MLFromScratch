import sys
sys.path.append('..')
import numpy as np
from .helpers import random_one_hot_matrix, random_stochastic_matrix,random_tensor, TFNCELoss, PyTorch_LSTM_many2many
from numpy_ml.deep_learning.loss_functions import BinaryCrossEntropy, SquaredLoss, NCELoss
from numpy.testing import assert_almost_equal
import torch.nn as nn
import torch
from numpy_ml.deep_learning.activation_functions import Sigmoid, Softmax, ReLU, LeakyReLU, TanH, FullSoftmax
from numpy_ml.deep_learning.layers import Dense, Embedding, BatchNormalization, RNNCell
from numpy_ml.deep_learning.layers import BidirectionalLSTM, many2oneRNN, LSTMCell, many2oneLSTM,DotProductAttention
from numpy_ml.deep_learning.optimizers import StochasticGradientDescent, Adagrad, RMSprop, Adadelta, Adam
from numpy_ml.deep_learning.schedulers import CosineAnnealingLR, CosineAnnealingWarmRestarts
from numpy_ml.utils import DiscreteSampler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

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

def test_square_loss(cases):
    np.random.seed(12346)

    cases = int(cases)

    mine = SquaredLoss()
    gold = nn.MSELoss(reduction='sum')

    # ensure we get 0 when the two arrays are equal
    n_classes = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_tensor((n_examples, n_classes), standardize=False)
    y_tensor = torch.tensor(y)

    y_pred_tensor = torch.tensor(y_pred)
    assert_almost_equal(mine.loss(y, y_pred), gold(y_tensor, y_pred_tensor))

    # test on random inputs
    i = 1

    while i < cases:
        n_classes = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 10)

        y = random_tensor((n_examples, n_classes), standardize=False)
        y_tensor = torch.tensor(y)

        y_pred = random_tensor((n_examples, n_classes), standardize=False)
        y_pred_tensor = torch.tensor(y_pred, requires_grad=True)
        y_pred_tensor.retain_grad()

        gold_value = gold(y_pred_tensor,y_tensor)
        gold_value.backward()
        # compare forward value
        assert_almost_equal(np.sum(mine.loss(y, y_pred)), gold_value.detach().numpy())

        # compare backward value
        assert_almost_equal(mine.gradient(y, y_pred), y_pred_tensor.grad)

        i += 1
    print ('Successfully testing squared loss function!')

def test_NCE_loss(cases):

    np.random.seed(12345)

    N = int(cases)
    DECIMAL=4
    LR = 0.01
    MOMENTUM = 0
    nesterov = False

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 10)
        n_c = np.random.randint(1, 10)
        n_out = np.random.randint(1, 300)
        vocab_size = np.random.randint(200, 1000)
        num_negative_samples = np.random.randint(1, 10)

        # the output of the embedding layer
        embeddings = random_tensor((n_ex, n_c, n_out), standardize=True)
        target = np.random.randint(0, vocab_size, (n_ex, 1))

        # initialize probs
        probs = np.random.rand(vocab_size)
        probs /= probs.sum()

        # initialize the discrete sampler
        D = DiscreteSampler(probs, log=False, with_replacement=False)
        NCE = NCELoss(n_classes=vocab_size,
                      n_in=n_out,
                      noise_sampler=D,
                      num_negative_samples=num_negative_samples,
                      trainable=False)
        mine_optim = StochasticGradientDescent(learning_rate=LR,momentum=MOMENTUM,nesterov=nesterov)
        NCE.initialize(mine_optim)
        mine_loss, _ = NCE.loss(embeddings, target.flatten(), None, True)
        mine_dLdX = NCE.gradient()
        mine_dLdW = NCE.dW
        mine_dLdb = NCE.db

        gold_loss = 0

        gold_dLdX = np.zeros_like(embeddings)
        gold_dLdW = np.zeros_like(NCE.W)
        gold_dLdb = np.zeros_like(NCE.b)
        # nv = (neg_samples, p_target, p_neg_samples)
        nv = NCE.derived_variables['noise_samples'][0]
        for ix, emb in enumerate(embeddings):
            sv = (nv[0], np.array([nv[1][0, ix]]), nv[2])

            NCE.X = []
            for k, v in NCE.derived_variables.items():
                NCE.derived_variables[k] = []

            NCE.dW = np.zeros_like(NCE.W)
            NCE.db = np.zeros_like(NCE.b)

            mine_this_loss, _ = NCE.loss(emb[None, :, :], target[ix], neg_samples=sv[0])
            NCE.derived_variables["noise_samples"] = [sv]
            dldx = NCE.gradient()
            NCE.derived_variables["noise_samples"] = sv

            TF_dict = TFNCELoss(emb, np.array([target[ix]]), NCE)

            this_gold_loss = TF_dict["final_loss"]
            gold_loss += this_gold_loss

            gold_dLdX[ix, ...] += TF_dict["dLdX"]

            gold_dLdW[:, TF_dict["dLdW"].indices] += TF_dict["dLdW"].values.transpose()
            gold_dLdb[:, TF_dict["dLdb"].indices] += TF_dict["dLdb"].values


        # compare forward process
        assert_almost_equal(mine_loss, gold_loss, decimal=DECIMAL)
        # compare backward process
        assert_almost_equal(mine_dLdX, gold_dLdX, decimal=DECIMAL)
        assert_almost_equal(mine_dLdW, gold_dLdW, decimal=DECIMAL)
        assert_almost_equal(mine_dLdb, gold_dLdb, decimal=DECIMAL)

        i += 1
    print ("Successfully testing Noise Contrastive Estimation loss function!")

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

    print ('Successfully testing softmax with cross entropy function!')

def test_full_softmax_activation(cases):

    N = int(cases)
    np.random.seed(12345)
    DECIMAL=5
    i = 0
    while i < N:
        n_ex = np.random.randint(1, 100)
        n_dims = np.random.randint(1, 100)
        z = random_tensor((n_ex, n_dims))
        z_tensor = torch.tensor(z,requires_grad=True)
        z_tensor.retain_grad()

        mine = FullSoftmax()
        gold = nn.Softmax(dim=1)

        gold_value = gold(z_tensor)
        gold_value.retain_grad()

        loss_tensor = torch.square(gold_value).sum()/2.
        loss_tensor.backward()

        gold_grad = z_tensor.grad
        mine_value = mine(z)
        mine_grad = mine.gradient(z, mine_value)

        # compare forward
        assert_almost_equal(mine_value, gold_value.detach().numpy(),decimal=DECIMAL)
        # compare backward
        assert_almost_equal(mine_grad, gold_grad, decimal=DECIMAL)
        i += 1
    print ('Successfully testing full softmax function!')


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
        mine = Dense(n_units = n_out, input_shape=(n_ex,n_in))
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

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

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

def test_Adagrad(cases):
    """
    The idea is to do one epoch training and the compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    LR = 0.05
    EPOCHS = 2

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)

        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
        mine_optim = Adagrad(learning_rate=LR)

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

        for this_epoch in range(EPOCHS):
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
    print ('Successfully testing Adagrad optimizer!')

def test_RMSprop(cases):
    """
    The idea is to do one epoch training and the compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    LR = 0.05
    ALPHA=0.99
    EPOCHS = 2

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)

        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=ALPHA)
        mine_optim = RMSprop(learning_rate=LR,rho=ALPHA)

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

        for this_epoch in range(EPOCHS):
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
    print('Successfully testing RMSprop optimizer!')

def test_Adadelta(cases):
    """
    The idea is to do one epoch training and the compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    ALPHA=0.9
    EPOCHS = 2

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)

        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        optimizer = torch.optim.Adadelta(model.parameters(), rho=ALPHA)
        mine_optim = Adadelta(rho=ALPHA)

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

        for this_epoch in range(EPOCHS):
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
    print('Successfully testing Adadelta optimizer!')


def test_Adam(cases):
    """
    The idea is to do one epoch training and the compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    b1 = 0.9
    b2 = 0.99
    LR = 0.05
    EPOCHS = 2

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)

        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(b1, b2))
        mine_optim = Adam(learning_rate=LR, b1=b1, b2=b2)

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

        for this_epoch in range(EPOCHS):
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
    print('Successfully testing Adam optimizer!')

def test_RNNCell(cases):

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
        gold = nn.RNNCell(input_size=n_in, hidden_size=n_out, bias=True)
        mine = RNNCell(n_units = n_out, activation='tanh',input_shape=(n_ex,n_in))
        # Do not allow the weights to be updated
        mine.trainable=False
        mine.initialize(None)

        # Adjust parameters to make them share the same set of weights and bias
        mine.W_i = gold.weight_ih.detach().numpy().transpose()
        mine.b_i = gold.bias_ih.detach().numpy()[None,:]

        mine.W_p = gold.weight_hh.detach().numpy().transpose()
        mine.b_p = gold.bias_hh.detach().numpy()[None,:]

        # forward prop
        gold_value = gold(X_tensor)
        mine_value = mine.forward_pass(X)


        # loss
        gold_loss = torch.square(gold_value).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdwi = gold.weight_ih.grad.detach().numpy()
        gold_dLdbi = gold.bias_ih.grad.detach().numpy()
        gold_dLdWp = gold.weight_hh.grad.detach().numpy()
        gold_dLdbp = gold.bias_hh.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        dLdX,_ = mine.backward_pass(mine_value)
        dLdWi = mine.dW_i
        dLdbi = mine.db_i
        dLdWp = mine.dW_p
        dLdbp = mine.db_p


        # compare forward
        assert_almost_equal(mine_value,gold_value.detach().numpy(),decimal=decimal)
        # compare backward
        assert_almost_equal(dLdWi.transpose(), gold_dLdwi,decimal=decimal)
        assert_almost_equal(dLdbi, gold_dLdbi[None,:],decimal=decimal)
        assert_almost_equal(dLdWp.transpose(), gold_dLdWp,decimal=decimal)
        assert_almost_equal(dLdbp, gold_dLdbp[None,:],decimal=decimal)
        assert_almost_equal(dLdX, gold_dLdX,decimal=decimal)
        i += 1
    print ("Successfully testing single RNN cell!")

def test_RNN_many2one(cases):

    np.random.seed(12345)
    N = int(cases)
    decimal = 5
    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)
        X_tensor = torch.tensor(X, dtype=torch.float, requires_grad=True)

        # initialize FC layer
        gold = nn.RNNCell(input_size=n_in, hidden_size=n_out, bias=True)
        mine = many2oneRNN(n_units = n_out, activation='tanh',input_shape=(n_ex,n_in),trainable=False)
        # Do not allow the weights to be updated
        mine.initialize(None)

        # Adjust parameters to make them share the same set of weights and bias
        mine.cell.W_i = gold.weight_ih.detach().numpy().transpose()
        mine.cell.b_i = gold.bias_ih.detach().numpy()[None,:]

        mine.cell.W_p = gold.weight_hh.detach().numpy().transpose()
        mine.cell.b_p = gold.bias_hh.detach().numpy()[None,:]

        # forward prop
        gold_hidden_val = [] # gold_hidden_val is used to compare the forward progress
        gold_value = [] # gold_value is used to compare backward progress
        ht = torch.tensor(np.zeros((n_ex, n_out)), dtype=torch.float, requires_grad=True)
        ht.retain_grad()
        for this_t in range(n_t):
            gold_value += [ht]
            gold_value_t = gold(input=X_tensor[:,:,this_t], hx=ht)
            gold_hidden_val.append(gold_value_t.detach().numpy())
            ht.retain_grad()
            ht = gold_value_t

        ht.retain_grad()
        gold_value += [ht]

        gold_hidden_val = np.dstack(gold_hidden_val)
        mine_value = mine.forward_pass(X)

        # loss
        gold_loss = torch.square(gold_value[-1]).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdwi = gold.weight_ih.grad.detach().numpy()
        gold_dLdbi = gold.bias_ih.grad.detach().numpy()
        gold_dLdWp = gold.weight_hh.grad.detach().numpy()
        gold_dLdbp = gold.bias_hh.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        # we will do many to one backpropagation, considering the loss function the input is mine_value[:,:,-1]
        dLdX = mine.backward_pass(mine_value[:,:,-1])
        dLdWi = mine.cell.dW_i
        dLdbi = mine.cell.db_i
        dLdWp = mine.cell.dW_p
        dLdbp = mine.cell.db_p


        # compare forward
        assert_almost_equal(mine_value,gold_hidden_val,decimal=decimal)
        # compare backward
        assert_almost_equal(dLdWi.transpose(), gold_dLdwi,decimal=decimal)
        assert_almost_equal(dLdbi, gold_dLdbi[None,:],decimal=decimal)
        assert_almost_equal(dLdWp.transpose(), gold_dLdWp,decimal=decimal)
        assert_almost_equal(dLdbp, gold_dLdbp[None,:],decimal=decimal)

        for this_t in range(n_t):
            # compare dLdX
            assert_almost_equal(dLdX[:,:,this_t], gold_dLdX[:,:,this_t],decimal=decimal)
            # compare dLdhidden_state
            assert_almost_equal(mine.cell.derived_variables['dLdA_accumulator'][this_t],
                                gold_value[this_t].grad,
                                decimal=decimal)
        i += 1
    print ('Successfully testing RNN many2one layer!')

def test_LSTMCell(cases):

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
        gold = nn.LSTMCell(input_size=n_in, hidden_size=n_out, bias=True)
        mine = LSTMCell(n_units = n_out, input_shape=(n_ex,n_in))
        # Do not allow the weights to be updated
        mine.trainable=False
        mine.initialize(None)


        # Adjust parameters to make them share the same set of weights and bias
        # Reference on the the shape of weight in PyTorch:
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        mine.W_ih = gold.weight_ih.detach().numpy().transpose()
        mine.W_hh = gold.weight_hh.detach().numpy().transpose()

        mine.b_ih = gold.bias_ih.detach().numpy()[None,:]
        mine.b_hh = gold.bias_hh.detach().numpy()[None,:]

        # forward prop
        gold_hidden, gold_cell = gold(X_tensor)
        mine_hidden, mine_cell = mine.forward_pass(X)

        # loss
        gold_loss = torch.square(gold_hidden).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdwi = gold.weight_ih.grad.detach().numpy()
        gold_dLdbi = gold.bias_ih.grad.detach().numpy()
        gold_dLdWh = gold.weight_hh.grad.detach().numpy()
        gold_dLdbh = gold.bias_hh.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        dLdX,_ = mine.backward_pass(mine_hidden)
        dLdWi = mine.dW_ih
        dLdbi = mine.db_ih
        dLdWh = mine.dW_hh
        dLdbh = mine.db_hh


        # compare forward
        assert_almost_equal(mine_hidden,gold_hidden.detach().numpy(),decimal=decimal)
        assert_almost_equal(mine_cell,gold_cell.detach().numpy(),decimal=decimal)
        # compare backward
        assert_almost_equal(dLdWi.transpose(), gold_dLdwi,decimal=decimal)
        assert_almost_equal(dLdbi, gold_dLdbi[None,:],decimal=decimal)
        assert_almost_equal(dLdWh.transpose(), gold_dLdWh,decimal=decimal)
        assert_almost_equal(dLdbh, gold_dLdbh[None,:],decimal=decimal)
        assert_almost_equal(dLdX, gold_dLdX,decimal=decimal)
        i += 1
    print ("Successfully testing a single LSTM cell!")

def test_LSTM_many2one(cases):

    np.random.seed(12345)
    N = int(cases)
    decimal = 5
    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)
        X_tensor = torch.tensor(X, dtype=torch.float, requires_grad=True)

        # initialize FC layer
        gold = nn.LSTMCell(input_size=n_in, hidden_size=n_out, bias=True)
        mine = many2oneLSTM(n_units = n_out,input_shape=(n_ex,n_in),trainable=False)
        # Do not allow the weights to be updated
        mine.initialize(None)

        # Adjust parameters to make them share the same set of weights and bias
        mine.cell.W_ih = gold.weight_ih.detach().numpy().transpose()
        mine.cell.b_ih = gold.bias_ih.detach().numpy()[None,:]

        mine.cell.W_hh = gold.weight_hh.detach().numpy().transpose()
        mine.cell.b_hh = gold.bias_hh.detach().numpy()[None,:]

        # forward prop
        gold_hidden_val = [] # gold_hidden_val is used to compare the forward progress
        gold_cell_val = []
        gold_hidden_grad = [] # gold_hidden_grad is used to compare backward progress
        gold_cell_grad = []

        ht = torch.tensor(np.zeros((n_ex, n_out)), dtype=torch.float, requires_grad=True)
        ht.retain_grad()

        ct = torch.tensor(np.zeros((n_ex, n_out)), dtype=torch.float, requires_grad=True)
        ct.retain_grad()

        for this_t in range(n_t):
            gold_hidden_grad += [ht]
            gold_cell_grad += [ct]

            gold_h, gold_c = gold(input=X_tensor[:,:,this_t],hx=(ht,ct))

            gold_hidden_val.append(gold_h.detach().numpy())
            gold_cell_val.append(gold_c.detach().numpy())

            ht.retain_grad()
            ct.retain_grad()
            ht = gold_h
            ct = gold_c

        ht.retain_grad()
        ct.retain_grad()
        gold_hidden_grad += [ht]
        gold_cell_grad += [ct]

        gold_hidden_val = np.dstack(gold_hidden_val)
        gold_cell_val = np.dstack(gold_cell_val)
        mine_hidden, mine_cell = mine.forward_pass(X)


        # loss
        gold_loss = torch.square(gold_hidden_grad[-1]).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdwi = gold.weight_ih.grad.detach().numpy()
        gold_dLdbi = gold.bias_ih.grad.detach().numpy()
        gold_dLdWh = gold.weight_hh.grad.detach().numpy()
        gold_dLdbh = gold.bias_hh.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        # we will do many to one backpropagation, considering the loss function the input is mine_value[:,:,-1]
        dLdX = mine.backward_pass(mine_hidden[:,:,-1])
        dLdWi = mine.cell.dW_ih
        dLdbi = mine.cell.db_ih
        dLdWh = mine.cell.dW_hh
        dLdbh = mine.cell.db_hh

        # compare forward
        assert_almost_equal(mine_hidden,gold_hidden_val,decimal=decimal)
        assert_almost_equal(mine_cell,gold_cell_val,decimal=decimal)
        # compare backward weights
        assert_almost_equal(dLdWi.transpose(), gold_dLdwi,decimal=decimal)
        assert_almost_equal(dLdbi, gold_dLdbi[None,:],decimal=decimal)
        assert_almost_equal(dLdWh.transpose(), gold_dLdWh,decimal=decimal)
        assert_almost_equal(dLdbh, gold_dLdbh[None,:],decimal=decimal)

        for this_t in range(n_t):
            # compare dLdX
            assert_almost_equal(dLdX[:,:,this_t], gold_dLdX[:,:,this_t],decimal=decimal)
            # compare dLdhidden_state
            assert_almost_equal(mine.cell.derived_variables['dLdA_prev'][this_t],
                                gold_hidden_grad[this_t].grad,
                                decimal=decimal)
            # compare dLdMemory_value
            assert_almost_equal(mine.cell.derived_variables['dLdC_prev'][this_t],
                                gold_cell_grad[this_t+1].grad,
                                decimal=decimal)

        i += 1
    print ("Successfully testing LSTM many2one layer!")

def test_LSTM_bidirection(cases):

    np.random.seed(12345)
    N = int(cases)
    decimal = 5
    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        n_t = np.random.randint(1, 10)
        X = random_tensor((n_ex, n_in, n_t), standardize=True)
        X_tensor = torch.tensor(X, dtype=torch.float, requires_grad=True).permute(2,0,1)
        X_tensor.retain_grad()

        # initialize FC layer
        gold = nn.LSTM(input_size=n_in, hidden_size=n_out, bias=True, num_layers=1,bidirectional=True)
        mine = BidirectionalLSTM(n_units = n_out,input_shape=(n_ex,n_in),trainable=False)
        # Do not allow the weights to be updated
        mine.initialize(None)

        # Adjust parameters to make them share the same set of weights and bias
        mine.cell_fwd.W_ih = gold.weight_ih_l0.detach().numpy().transpose()
        mine.cell_fwd.b_ih = gold.bias_ih_l0.detach().numpy()[None,:]
        mine.cell_fwd.W_hh = gold.weight_hh_l0.detach().numpy().transpose()
        mine.cell_fwd.b_hh = gold.bias_hh_l0.detach().numpy()[None,:]

        mine.cell_bwd.W_ih = gold.weight_ih_l0_reverse.detach().numpy().transpose()
        mine.cell_bwd.b_ih = gold.bias_ih_l0_reverse.detach().numpy()[None,:]
        mine.cell_bwd.W_hh = gold.weight_hh_l0_reverse.detach().numpy().transpose()
        mine.cell_bwd.b_hh = gold.bias_hh_l0_reverse.detach().numpy()[None,:]


        # forward prop,
        # I canfirmed that for each direction, gold_h is the last element
        # of gold_output because gold_h is the hidden state at the last timestep
        # while gold_output includes hidden state values of all timesteps.
        gold_output, (gold_h, gold_c) = gold(input=X_tensor)
        # 2 is the number of direction
        gold_output = gold_output.view(n_t, n_ex, 2, n_out)
        # 1 is the number of layer, 2 is the number of directions
        gold_h = gold_h.view(1, 2, n_ex, n_out)
        gold_c = gold_c.view(1, 2, n_ex, n_out)


        mine_H_fwd, mine_H_bwd, mine_C_fwd, mine_C_bwd = mine.forward_pass(X)

        # loss, this loss function is desgined that both directions can be tested
        # the loss function cannot solely come from the forward direction or the backward direction, otherwise
        # PyTorch cannot calculate the gradients for the other direction.
        gold_loss = torch.square(gold_output[-1,:,0,:]).sum()/2. + torch.square(gold_output[0,:,1,:]).sum()/2.
        gold_loss.backward()

        # backprop
        gold_dLdwi_fwd = gold.weight_ih_l0.grad.detach().numpy()
        gold_dLdbi_fwd = gold.bias_ih_l0.grad.detach().numpy()
        gold_dLdWh_fwd = gold.weight_hh_l0.grad.detach().numpy()
        gold_dLdbh_fwd = gold.bias_hh_l0.grad.detach().numpy()

        gold_dLdwi_bwd = gold.weight_ih_l0_reverse.grad.detach().numpy()
        gold_dLdbi_bwd = gold.bias_ih_l0_reverse.grad.detach().numpy()
        gold_dLdWh_bwd = gold.weight_hh_l0_reverse.grad.detach().numpy()
        gold_dLdbh_bwd = gold.bias_hh_l0_reverse.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()


        dLdX_fwd, dLdX_bwd = mine.backward_pass(mine_H_fwd[:,:,-1], mine_H_bwd[:,:,0])

        dLdWi_fwd = mine.cell_fwd.dW_ih
        dLdbi_fwd = mine.cell_fwd.db_ih
        dLdWh_fwd = mine.cell_fwd.dW_hh
        dLdbh_fwd = mine.cell_fwd.db_hh

        dLdWi_bwd = mine.cell_bwd.dW_ih
        dLdbi_bwd = mine.cell_bwd.db_ih
        dLdWh_bwd = mine.cell_bwd.dW_hh
        dLdbh_bwd = mine.cell_bwd.db_hh

        # compare forward pass
        for this_t in range(n_t):
            # forward direction: hidden state
            assert_almost_equal(mine_H_fwd[:, :, this_t], gold_output[this_t,:,0,:].detach().numpy(),decimal=decimal)
            # backward direction: hidden state
            assert_almost_equal(mine_H_bwd[: , :, this_t], gold_output[this_t,:,1,:].detach().numpy(),decimal=decimal)
        # we only compare the cell value at the last timestep because PyTorch only returns the memory cell value
        # of the last timestep.
        assert_almost_equal(mine_C_fwd[:, :, -1], gold_c[0,0,:,:].detach().numpy(),decimal=decimal)
        assert_almost_equal(mine_C_bwd[:, :, 0], gold_c[0,1,:,:].detach().numpy(),decimal=decimal)

        # compare backward weights
        assert_almost_equal(dLdWi_fwd.transpose(), gold_dLdwi_fwd,decimal=decimal)
        assert_almost_equal(dLdbi_fwd, gold_dLdbi_fwd[None,:],decimal=decimal)
        assert_almost_equal(dLdWh_fwd.transpose(), gold_dLdWh_fwd,decimal=decimal)
        assert_almost_equal(dLdbh_fwd, gold_dLdbh_fwd[None,:],decimal=decimal)

        assert_almost_equal(dLdWi_bwd.transpose(), gold_dLdwi_bwd,decimal=decimal)
        assert_almost_equal(dLdbi_fwd, gold_dLdbi_fwd[None,:],decimal=decimal)
        assert_almost_equal(dLdWh_fwd.transpose(), gold_dLdWh_fwd,decimal=decimal)
        assert_almost_equal(dLdbh_fwd, gold_dLdbh_fwd[None,:],decimal=decimal)


        for this_t in range(n_t):
            # compare dLdX
            assert_almost_equal(dLdX_fwd[:,:,this_t]+dLdX_bwd[:,:,this_t], gold_dLdX[this_t],decimal=decimal)

        i += 1
    print ("Successfully testing bidirectional LSTM layer!")


def test_LSTM_many2many(cases):
    np.random.seed(12345)
    N = int(cases)
    decimal = 5
    i = 1
    gold_criterion=nn.MSELoss(reduction='sum')
    mine_criterion = SquaredLoss()
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        n_t = np.random.randint(1, 10)

        # initialize input and target
        X = random_tensor((n_ex, n_in, n_t), standardize=True)
        X_tensor = torch.tensor(X, dtype=torch.float64, requires_grad=True).permute(2,0,1)
        X_tensor.retain_grad()

        target = random_tensor((n_t, n_ex, 1), standardize=True)
        target_tensor=torch.tensor(target, dtype=torch.float64, requires_grad=True).double()

        # initialize model
        gold = PyTorch_LSTM_many2many(input_size=n_in, hidden_size=n_out)

        # TODO: wrap up two layers into one model
        mine_lstm = many2oneLSTM(n_units = n_out,input_shape=(n_ex, n_in),trainable=False)
        mine_lstm.initialize(None)

        mine_linear = Dense(n_units = 1, input_shape=(n_ex, n_out), trainable=False)
        mine_linear.initialize(None)

        # make sure both the gold and testing models share the same initial weights
        mine_lstm.cell.W_hh = gold.lstm.weight_hh_l0.detach().numpy().transpose()
        mine_lstm.cell.b_hh = gold.lstm.bias_hh_l0.detach().numpy()[None,:]

        mine_lstm.cell.W_ih = gold.lstm.weight_ih_l0.detach().numpy().transpose()
        mine_lstm.cell.b_ih = gold.lstm.bias_ih_l0.detach().numpy()[None,:]

        mine_linear.W = gold.linear.weight.detach().numpy().transpose()
        mine_linear.b = gold.linear.bias.detach().numpy()[None,:]

        # forward
        gold_prediction = gold(X_tensor)
        gold_loss = gold_criterion(gold_prediction, target_tensor)

        mine_lstm_hidden, mine_lstm_cell = mine_lstm.forward_pass(X)
        mine_final_output = []
        for this_t in range(n_t):
            mine_this_output = mine_linear.forward_pass(mine_lstm_hidden[:,:,this_t])
            mine_final_output.append(mine_this_output)
        mine_final_output = np.stack(mine_final_output)
        mine_loss = 0
        for this_t in range(n_t):
            mine_loss+=np.sum(mine_criterion.loss(target[this_t], mine_final_output[this_t]))

        #backward
        gold_loss.backward()
        gold_dLdW_ih_lstm = gold.lstm.weight_ih_l0.grad.detach().numpy()
        gold_dLdb_ih_lstm = gold.lstm.bias_ih_l0.grad.detach().numpy()
        gold_dLdW_hh_lstm = gold.lstm.weight_hh_l0.grad.detach().numpy()
        gold_dLdb_hh_lstm = gold.lstm.bias_hh_l0.grad.detach().numpy()
        gold_dLdW_linear = gold.linear.weight.grad.detach().numpy()
        gold_dLdb_linear = gold.linear.bias.grad.detach().numpy()
        gold_dLdX = X_tensor.grad.detach().numpy()

        mine_dLdX = np.zeros_like(X)
        for this_t in reversed(range(n_t)):
            mine_dLdlstm_hidden = mine_linear.backward_pass(-2*(target[this_t]-mine_final_output[this_t]))
            this_mine_dLdX = mine_lstm.backward_pass(mine_dLdlstm_hidden)
            last_dim = this_mine_dLdX.shape[2]
            while last_dim < n_t:
                this_mine_dLdX = np.insert(this_mine_dLdX, last_dim, 0, axis=2)
                last_dim += 1
            mine_dLdX += this_mine_dLdX

        # compare forward
        assert_almost_equal(mine_final_output, gold_prediction.detach().numpy(),decimal=decimal)
        assert_almost_equal(mine_loss, gold_loss.detach().numpy(),decimal=decimal)
        # compare backward
        assert_almost_equal(mine_linear.dw, gold_dLdW_linear.transpose(),decimal=decimal)
        assert_almost_equal(mine_linear.db, gold_dLdb_linear[None,...],decimal=decimal)
        assert_almost_equal(mine_lstm.cell.dW_hh, gold_dLdW_hh_lstm.transpose(),decimal=decimal)
        assert_almost_equal(mine_lstm.cell.db_ih, gold_dLdb_ih_lstm[None,...],decimal=decimal)
        assert_almost_equal(mine_lstm.cell.db_hh, gold_dLdb_hh_lstm[None,...],decimal=decimal)

        for this_t in range(n_t):
            assert_almost_equal(mine_dLdX[:,:,this_t], gold_dLdX[this_t],decimal=decimal)

        i += 1
    print ("Successfully testing LSTM many to many function!")

# embed_dim is the number of features
# In self-attention, the dimension of q,k,v are supposed to be the same in PyTorch. Actually, only the dimension of
# q and k have to be the same, v does not have to be equal. I don't understand why PyTorch requires the equality.
# In decoder, they cannot be the same. Is it for the sake of simplicity?
# add_zero_attn: https://github.com/pytorch/pytorch/issues/27461#issuecomment-656991245
def test_single_head_attention(cases):
    # ref: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L862
    np.random.seed(12345)
    N = int(cases)
    DECIMAL = 5

    i = 1
    gold_criterion=nn.MSELoss(reduction='sum')
    mine_criterion = SquaredLoss()
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_seq = np.random.randint(2, 100)
        d_q = np.random.randint(1, 100)
        X_emb = random_tensor((n_seq, n_ex, d_q), standardize=True)
        X_emb_tensor = torch.tensor(X_emb, dtype=torch.float64, requires_grad=True)

        target = random_tensor((n_seq, n_ex, d_q), standardize=True)
        target_tensor = torch.tensor(target, dtype=torch.float64, requires_grad=True)

        # initialize the single head attention layer
        gold = nn.MultiheadAttention(embed_dim=d_q,
                                     num_heads=1,
                                     dropout=0,
                                     bias=False,
                                     add_bias_kv=False,
                                     add_zero_attn=False,
                                     kdim=None,
                                     vdim=None).double()
        mine = DotProductAttention(emb_dim=d_q,
                                   d_k=None,
                                   d_v=None,
                                   trainable=False)
        mine.initialize(None)

        # make sure that they share the same weights
        mine.in_weight = gold.in_proj_weight.detach().numpy().transpose()
        mine.out_weight = gold.out_proj.weight.detach().numpy().transpose()

        # forward process
        gold_attn_output, gold_attn_output_weights = gold(query=X_emb_tensor,
                                                          key=X_emb_tensor,
                                                          value=X_emb_tensor,
                                                          need_weights=True)
        gold_attn_output.retain_grad()
        gold_attn_output_weights.retain_grad()
        mine_output, mine_weights, mine_scores = mine.forward_pass(X_emb, X_emb, X_emb)

        # calculate loss
        gold_loss = gold_criterion(gold_attn_output, target_tensor)
        mine_loss = np.sum(mine_criterion.loss(target, mine_output))

        # calculate gradients
        gold_loss.backward()
        gold_dLdout_weight = gold.out_proj.weight.grad.detach().numpy()
        gold_dLdin_weight = gold.in_proj_weight.grad.detach().numpy()
        gold_dLdX = X_emb_tensor.grad.detach().numpy()

        mine_dLdX = mine.backward_pass(-2*(target - mine_output))
        mine_dLdout_weight = mine.dLdout_weight
        mine_dLdin_weight = mine.dLdin_weight

        # compare forward process
        assert_almost_equal(mine_weights,
                            gold_attn_output_weights.detach().numpy(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_output,
                            gold_attn_output.detach().numpy(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_loss,
                            gold_loss.detach().numpy(),
                            decimal=DECIMAL)
        # compare backward process
        assert_almost_equal(mine_dLdout_weight,
                            gold_dLdout_weight.transpose(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_dLdX,
                            gold_dLdX,
                            decimal=DECIMAL)
        assert_almost_equal(mine_dLdin_weight,
                            gold_dLdin_weight.transpose(),
                            decimal=DECIMAL)

        i += 1
    print ("Successfully testing single head self-attention layer!")

def test_multi_head_attention(cases):
    # ref: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L862
    np.random.seed(12345)
    N = int(cases)
    DECIMAL = 5

    i = 1
    gold_criterion=nn.MSELoss(reduction='sum')
    mine_criterion = SquaredLoss()
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_seq = np.random.randint(2, 100)
        # I will fix the length of embedding to be 6 for multi head attention unit tests
        d_q = 6
        X_emb = random_tensor((n_seq, n_ex, d_q), standardize=True)
        X_emb_tensor = torch.tensor(X_emb, dtype=torch.float64, requires_grad=True)
        # I will fix the number of heads to be 3
        number_heads = 3

        target = random_tensor((n_seq, n_ex, d_q), standardize=True)
        target_tensor = torch.tensor(target, dtype=torch.float64, requires_grad=True)

        # initialize the single head attention layer
        gold = nn.MultiheadAttention(embed_dim=d_q,
                                     num_heads=number_heads,
                                     dropout=0,
                                     bias=False,
                                     add_bias_kv=False,
                                     add_zero_attn=False,
                                     kdim=None,
                                     vdim=None).double()
        mine = DotProductAttention(emb_dim=d_q,
                                   d_k=None,
                                   d_v=None,
                                   trainable=False,
                                   num_heads = number_heads)
        mine.initialize(None)

        # make sure that they share the same weights
        mine.in_weight = gold.in_proj_weight.detach().numpy().transpose()
        mine.out_weight = gold.out_proj.weight.detach().numpy().transpose()

        # forward process
        gold_attn_output, gold_attn_output_weights = gold(query=X_emb_tensor,
                                                          key=X_emb_tensor,
                                                          value=X_emb_tensor,
                                                          need_weights=True)
        gold_attn_output.retain_grad()
        mine_output, mine_weights, mine_scores = mine.forward_pass(X_emb, X_emb, X_emb)

        # calculate loss
        gold_loss = gold_criterion(gold_attn_output, target_tensor)
        mine_loss = np.sum(mine_criterion.loss(target, mine_output))

        # calculate gradients
        gold_loss.backward()
        gold_dLdout_weight = gold.out_proj.weight.grad.detach().numpy()
        gold_dLdin_weight = gold.in_proj_weight.grad.detach().numpy()
        gold_dLdX = X_emb_tensor.grad.detach().numpy()

        mine_dLdX = mine.backward_pass(-2*(target - mine_output))
        mine_dLdout_weight = mine.dLdout_weight
        mine_dLdin_weight = mine.dLdin_weight

        # compare forward process
        assert_almost_equal(mine_weights,
                            gold_attn_output_weights.detach().numpy(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_output,
                            gold_attn_output.detach().numpy(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_loss,
                            gold_loss.detach().numpy(),
                            decimal=DECIMAL)
        # compare backward process
        assert_almost_equal(mine_dLdout_weight,
                            gold_dLdout_weight.transpose(),
                            decimal=DECIMAL)
        assert_almost_equal(mine_dLdX,
                            gold_dLdX,
                            decimal=DECIMAL)
        assert_almost_equal(mine_dLdin_weight,
                            gold_dLdin_weight.transpose(),
                            decimal=DECIMAL)

        i += 1
    print ("Successfully testing multi head self-attention layer!")

def test_cosine_annealing_scheduler(cases):
    """
    The idea is to do one epoch training and then compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    LR = 0.05
    MOMENTUM = 0.9
    MIN_LR = 0.01 # Minimum learning rate
    T_MAX = 4 # Maximum number of iterations

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        epochs = np.random.randint(T_MAX, T_MAX*2)
        nesterov = np.random.choice(np.array([True,False]))
        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float64, requires_grad=True)

        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True).double()

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        gold_optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    nesterov=nesterov)
        gold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gold_optimizer,
                                                                    T_max=T_MAX,
                                                                    eta_min=MIN_LR,
                                                                    last_epoch=-1)
        gold_loss = torch.nn.MSELoss(reduction='sum')
        mine_loss = SquaredLoss()

        mine_scheduler = CosineAnnealingLR(min_lr=MIN_LR, t_max=T_MAX)
        mine_optim = StochasticGradientDescent(learning_rate=LR,momentum=MOMENTUM,nesterov=nesterov, scheduler=mine_scheduler)

        mine.trainable=True
        mine.initialize(mine_optim)

        mine.W = model.weight.detach().numpy().transpose().copy()
        mine.b = model.bias.detach().numpy()[None,:].copy()

        # generate the target variable
        Y = 5 * (X@mine.W.copy()+mine.b.copy()) + 10
        Y_tensor = torch.tensor(Y,dtype=torch.float64, requires_grad=True)

        # make sure initial weights are the same
        assert_almost_equal(mine.W, model.weight.detach().numpy().transpose(),decimal=decimal)
        assert_almost_equal(mine.b, model.bias.detach().numpy()[None,:],decimal=decimal)


        for this_epoch in range(epochs):
            gold_optimizer.zero_grad()
            # forward prop
            model_value = model(X_tensor)
            mine_value = mine.forward_pass(X)
            mine_loss_value = mine_loss.loss(Y,mine_value)
            model_loss = gold_loss(model_value, Y_tensor)
            # backward prop
            model_loss.backward()
            gold_optimizer.step()
            gold_scheduler.step()
            gold_weight = model.weight.detach().numpy()
            gold_bias = model.bias.detach().numpy()

            _ = mine.backward_pass(-2*(Y-mine_value))

            mine_weight = mine.W
            mine_bias = mine.b


            assert_almost_equal(mine_weight, gold_weight.transpose(),decimal=decimal)
            assert_almost_equal(mine_bias, gold_bias[None,:],decimal=decimal)

        i += 1
    print ("Successfully testing Cosine Annealing LR scheduler!")


def test_cosine_annealing_warm_restarts(cases):
    """
    The idea is to do one epoch training and then compare the weights and bias. This test depends on
    fully connected layers, and fully connected layer has been tested.
    """

    np.random.seed(12345)

    N = int(cases)

    decimal = 4
    LR = 0.05
    MOMENTUM = 0.9
    MIN_LR = 0.01 # Minimum learning rate
    T_0 = 4 # The initial maximum number of iterations
    T_MULT = 2

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        epochs = np.random.randint(T_0, T_0*T_MULT)
        nesterov = np.random.choice(np.array([True,False]))
        X = random_tensor((n_ex, n_in), standardize=True)
        X_tensor = torch.tensor(X,dtype=torch.float64, requires_grad=True)


        # initialize FC layer
        model = nn.Linear(in_features=n_in, out_features=n_out, bias=True).double()

        mine = Dense(n_units = n_out, input_shape=(n_ex, n_in))

        # initialize the SGD optimizer
        gold_optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    nesterov=nesterov)
        gold_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(gold_optimizer,
                                                                               T_0=T_0,
                                                                               T_mult=T_MULT,
                                                                               eta_min=MIN_LR,
                                                                               last_epoch=-1)
        gold_loss = torch.nn.MSELoss(reduction='sum')
        mine_loss = SquaredLoss()
        mine_scheduler = CosineAnnealingWarmRestarts(min_lr=MIN_LR, t_0=T_0, t_mult=T_MULT)
        mine_optim = StochasticGradientDescent(learning_rate=LR,momentum=MOMENTUM,nesterov=nesterov, scheduler=mine_scheduler)

        mine.trainable=True
        mine.initialize(mine_optim)

        mine.W = model.weight.detach().numpy().transpose().copy()
        mine.b = model.bias.detach().numpy()[None,:].copy()

        # generate the target variable
        Y = 5 * (X@mine.W.copy()+mine.b.copy()) + 10
        Y_tensor = torch.tensor(Y,dtype=torch.float64, requires_grad=True)

        # make sure initial weights are the same
        assert_almost_equal(mine.W, model.weight.detach().numpy().transpose(),decimal=decimal)
        assert_almost_equal(mine.b, model.bias.detach().numpy()[None,:],decimal=decimal)


        for this_epoch in range(epochs):
            gold_optimizer.zero_grad()
            # forward prop
            model_value = model(X_tensor)
            mine_value = mine.forward_pass(X)
            mine_loss_value = mine_loss.loss(Y,mine_value)
            model_loss = gold_loss(model_value, Y_tensor)
            # backward prop
            model_loss.backward()
            gold_optimizer.step()
            gold_scheduler.step()

            gold_weight = model.weight.detach().numpy()
            gold_bias = model.bias.detach().numpy()

            _ = mine.backward_pass(-2*(Y-mine_value))

            mine_weight = mine.W
            mine_bias = mine.b

            assert_almost_equal(mine_weight, gold_weight.transpose(),decimal=decimal)
            assert_almost_equal(mine_bias, gold_bias[None,:],decimal=decimal)
        i += 1
    print ("Successfully testing Cosine Annealing Warm Restarts LR scheduler!")
