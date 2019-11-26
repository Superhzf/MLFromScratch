import numpy as np

"""
Example of nn_architecture:
input: n x m
n: n variables
m: m samples
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]
"""

def init_layers(nn_architecture,seed = 99):
    np.random.seed(seed)
    num_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer['output_dim']
        params_values[f'W_{layer_idx}'] = np.random.randn(layer_output_size,layer_input_size) * 0.1 # this could be wrong
        params_values[f'b_{layer_idx}'] = np.random.randn(layer_output_size,1) * 0.10

    return params_values
