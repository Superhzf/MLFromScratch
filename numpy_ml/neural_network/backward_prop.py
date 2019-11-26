import numpy as np
import sys
sys.path.insert(0, '')
from activations import sigmoid_backward,ReLU_backward


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation = 'relu'):
    m = A_prev.shape[1]

    if activation == 'relu':
        backward_activation_func = ReLU_backward
    elif activation == 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / m # is it correct?
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def fully_backward_propagation(y_hat, y, memory, params_values, nn_architecture):
    y_hat = np.array(y_hat)
    y = np.array(y)
    grads_values = {}
    m = len(y_hat)

    dA_prev = -y/(y_hat+1e-15)+(1-y)/(1-y_hat+1e-15)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev+1
        activation_func_curr = layer['activation']

        dA_curr = dA_prev
        A_prev = memory[f'A_{layer_idx_prev}']
        Z_curr = memory[f'Z_{layer_idx_curr}']
        W_curr = params_values[f'W_{layer_idx_curr}']
        b_curr = params_values[f'b_{layer_idx_curr}']

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(dA_curr,W_curr,b_curr,Z_curr, A_prev,activation_func_curr)

        grads_values[f'dW_{layer_idx_curr}'] = dW_curr
        grads_values[f'db_{layer_idx_curr}'] = db_curr

    return grads_values