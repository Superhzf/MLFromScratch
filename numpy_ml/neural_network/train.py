# ref: https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
# https://explained.ai/matrix-calculus/
# http://cs231n.stanford.edu/vecDerivs.pdf
import sys
sys.path.insert(0, '')
from initializers import init_layers
from forward_prop import fully_forward_propagation
from backward_prop import fully_backward_propagation
from losses import CrossEntropyLoss,get_accuracy_value
from updates import update

# TODO: early stop, mini_batch, normalization


def fit(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, seed=2)
    loss_history = []
    accuracy_history = []
    grads_history = []

    for i in range(epochs):
        y_hat, cache = fully_forward_propagation(X, params_values, nn_architecture)
        loss = CrossEntropyLoss(y_hat, Y)
        loss_history.append(loss)
        accuracy = get_accuracy_value(y_hat,Y)
        accuracy_history.append(accuracy)

        grads_values = fully_backward_propagation(y_hat, Y, cache, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        grads_history.append(grads_values)
    return params_values, loss_history, accuracy_history, grads_history
