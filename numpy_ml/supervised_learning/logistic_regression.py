import numpy as np
import math
from deep_learning.activation_functions import Sigmoid

class LogisticRegression():
    """
    Logistic Regression Classifier
    -------------------------------------
    learning rate: float
        The step length that will be taken when following the negative gradient
        during training
    """
    def __init__(self,learning_rate=0.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self,X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N),1/sqrt(N)]
        limit = 1/math.sqrt(n_features)
        self.param = np.random.uniform(-limit,limit,(n_features,))

    def fit(self,X,y,n_iterations = 1000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            self.param =  self.param - self.learning_rate*((y_pred-y).dot(X))

    def predict(self,X):
        y_pred = self.sigmoid(X.dot(self.param)) # raw prediction
        return y_pred
