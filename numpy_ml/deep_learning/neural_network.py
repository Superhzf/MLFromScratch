import numpy as np
import progressbar
from 

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

        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self,X,y):
        """Evaluates the model over a single batch of samples"""
        y_pred = self._forward_pass(X,training=False)
        loss = np.mean(self.loss_function.loss(y,y_pred))
        acc = self.loss_function.acc(y,y_pred)

        return loss, acc


    def train_on_batch(self,X,y):
        """Single gradient update over one batch of samples"""
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y,y_pred))
        acc = self.loss_function.acc(y,y_pred)
        # Calculate the gradient of the loss function w.r.t y_pred
        loss_grad = self.loss_function.gradient(y,y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad = loss_grad)


    def fit(self,X,y,n_epochs,batch_size):
        """Train the model for a fixed number of epochs"""
        for _ in self.progressbar(range(n_epochs)):
            batch_error = []
            for X_batch,y_batch in batch
