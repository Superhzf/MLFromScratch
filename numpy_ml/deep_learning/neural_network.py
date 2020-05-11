from __future__ import print_function, division
from terminaltables import AsciiTable
from ..utils import batch_generator
from ..utils.misc import bar_widgets
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
        self.errors = {"training":[],"validation":[]}
        self.layers = []
        self.loss_function = loss()
        self.optimizer = optimizer
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data is not None:
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
        if self.layers and self.layers[-1].output_shape() != (None, ):
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

        return loss

    # why use batch gradient descent?
    # Answer: If the whole dataset is used, then the memory might be not large enough
    # If we use one sample each time, the training may go astray and we cannot
    # take the advantage of vectorization
    def train_on_batch(self,X,y):
        """Single gradient update over one batch of samples"""
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.loss(y,y_pred))
        # Calculate the gradient of the loss function w.r.t y_pred
        loss_grad = self.loss_function.gradient(y,y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad = loss_grad)
        return loss

    def fit(self,X,y,n_epochs,batch_size,print_loss_every_epoch=100):
        """Train the model for a fixed number of epochs"""
        for epoch in self.progressbar(range(n_epochs)):
            batch_error = []
            if isinstance(X, np.ndarray):
                for X_batch,y_batch in batch_generator(X,y,batch_size = batch_size):
                    loss = self.train_on_batch(X_batch,y_batch)
                    batch_error.append(loss)
                if epoch % print_loss_every_epoch == 0:
                    print ('epoch:',epoch,'loss:',loss)
            elif isinstance(X, list):
                for X_batch, y_batch in zip(X,y):
                    if len(X_batch.shape) == 1:
                        X_batch = X_batch[None, :]
                    loss = self.train_on_batch(X_batch,y_batch)
                    batch_error.append(loss)
                if epoch % print_loss_every_epoch == 0:
                    print ('epoch:',epoch,'loss:',loss)

            self.errors['training'].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss,_ = self.test_on_batch(self.val_set['X'],self.val_set['y'])
                self.errors['validation'].append(val_loss)
        print ('final loss', loss)
        if self.val_set is not None:
            return self.errors['training'],self.errors['validation']
        else:
            return self.errors['training']


    def _forward_pass(self,X,training=True):
        """Calculate the output of NN"""
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)
        return layer_output


    def _backward_pass(self,loss_grad):
        """Propagage the gradient backwards and update the weights in each layer"""
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)


    def summary(self,name="Model Summary"):
        # Print model name
        print (AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print ("Input Shape: {}".format(self.layers[0].input_shape))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params+=params
        # Print network configuration table
        print (AsciiTable(table_data).table)
        print ("Total Parameters: {}\n".format(tot_params))


    def predict(self, X, fixed_len=True):
        """
        Use the trained model to predict labels of X. fixed_len==False means
        that the input length is variable and we cannot take the advantage of
        vectorization.
        """

        if fixed_len:
            return self._forward_pass(X, training=False)
        else:
            res = []
            for this_x in X:
                if len(this_x.shape) == 1:
                    this_x = this_x[None, :]
                this_pred = self._forward_pass(this_x, training=False)
                res.append(this_pred)
            return res
