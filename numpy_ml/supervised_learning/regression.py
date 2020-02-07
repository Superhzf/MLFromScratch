import numpy as np

class l1_regularization():
    """
    Regularization for Lasso Regression
    """
    def __init__(self,alpha):
        self.alpha = alpha

    def __call__(self,w):
        return self.alpha*np.linalg.norm(w,ord=1)

    def grad(self,w):
        return self.alpha * sign(w)

class l2_regularization():
    """
    Regularization for Ridge Regression
    """
    def __init__(self,alpha):
        self.alpha = alpha

    def __call__(self,w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def grad(self,w):
        return self.alpha * w

class l1_l2_regression():
    """
    Regularization for Elastic Net Regression
    """
    def __init__(self,alpha,l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self,w):
        l1_contr = self.l1_ratio*np.linalg.norm(w,ord=1)
        l2_contr = (1-self.l1_ratio)*0.5*w.T.dot(w)
        return self.alpha*(l1_contr+l2_contr)

    def grad(self,w):
        l1_contr = self.l1_ratio*np.sign(w)
        l2_contr = (1-self.l1_ratio)*w
        return self.alpha*(l1_contr+l2_contr)


class Regression(object):
    """
    Base regression class. This class should not be called by users

    Parameters
    ------------------------------
    n_iterations: float
        The number of training iterations the algorithm will tune the wights
    learning_rate: float
        The step length that will be used when updating the weights
    """
    def __init__(self,n_iterations,learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self,n_features):
        """
        Initialize weights randomly [-1/N,-1/N]
        """
        limit=1/np.sqrt(n_features)
        self.w = np.random.uniform(-limit,limit,(n_features,))

    def fit(self,X,y):
        # Insert constant ones for bias weights as the first column
        X = np.insert(X,0,1,axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # calculate l2 loss
            mse = np.mean(0.5*(y-y_pred)**2+self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = -(y-y_pred).dot(X) + self.regularization.grad(self.w)
            # update weights
            self.w -= self.learning_rate*grad_w

    def predict(self,X):
        X = np.insert(X,0,1,axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """
    Linear model

    Parameters
    -----------------------------
    n_iterations: float
        The number of training iterations the algorithm will go through
    learning_rate: float
        The step length that will be used when updating weights
    gradient_descent: boolean
        True: gradient descent will be used to update weights
        False: OLS will be used to update weights
    """
    def __init__(self,n_iterations=100,learning_rate=0.01,gradient_descent=True):
        self.gradient_descent=gradient_descent
        # No regularization
        self.regularization=lambda x:0
        self.regularization.grad=lambda x:0
        super(LinearRegression,self).__init__(n_iterations=n_iterations,
                                              learning_rate=learning_rate)

    def fit(self,X,y):
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X,0,1,axis=1)
            # Calculate weights by least squares ()
