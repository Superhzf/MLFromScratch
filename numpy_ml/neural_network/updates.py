import numpy as np

def update(params_values,grads_values, nn_architecture, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values[f'W_{layer_idx}'] -= learning_rate*grads_values[f'dW_{layer_idx}']
        params_values[f'b_{layer_idx}'] -= learning_rate*grads_values[f'db_{layer_idx}']

    return params_values