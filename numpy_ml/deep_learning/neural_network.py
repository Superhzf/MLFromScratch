import numpy as np
import progressbar

class NeuralNetwork():
    """
    Deep learning base model

    Parameters:
    ----------------------
    optimizer: class
        The weight optimizer that will be used to tune weights in order to
        optimize the loss
    loss: class
        Loss function used to measure the model's performance. SquareLoss or
        CrossEntropy

        A tuple containing validation data and labels (X,y)
    """
    def __init__(self,optimizer,loss,validation_data = None):
        self.errors = {"training:"[],"validation":[]}
        self.layers = []
        self.loss_function = loss()
        self.optimizer = optimizer
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {'X':X,'y':y}


    def set_trainable(self,trainable):
        """
        Method which enables freezing of the weights of the network's layers
        """
        for layer in self.layers:
            layer.trainable = trainable


    def add(self,layer):
        """Method which adds a layer to the neural network"""
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape = self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized
        if hasattr(layer,'initialize'):
            layer.initialize(optimizer=self.optimizer)
