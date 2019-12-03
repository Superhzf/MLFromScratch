import numpy as np


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A


def ReLU(Z):
    A = np.maximum(0, Z)
    return A


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    dZ = dA * sig * (1 - sig)
    return dZ


def ReLU_backward(dA,Z):
    dZ = np.array(dA)
    dZ[Z < 0] = 0
    return dZ
