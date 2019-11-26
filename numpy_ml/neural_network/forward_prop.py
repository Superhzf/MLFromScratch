import sys
sys.path.insert(0, '')
from activations import ReLU,sigmoid


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation = 'relu'):
    Z_curr = W_curr @ A_prev + b_curr
    if activation == 'relu':
        activation_func = ReLU
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        return Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def fully_forward_propagation(X, param_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activation_func_curr = layer['activation']
        W_curr = param_values[f'W_{layer_idx}']
        b_curr = param_values[f'b_{layer_idx}']
        A_curr, Z_curr = single_layer_forward_propagation(A_prev,W_curr,b_curr,activation_func_curr)

        memory[f'A_{idx}'] = A_prev
        memory[f'Z_{layer_idx}'] = Z_curr

    return A_curr, memory

