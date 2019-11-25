import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def ReLU(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA,Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def ReLU_backward(dA,Z):
    dZ = np.array(dA,copy=True)
    dZ[Z<0] = 0
    return dZ