import numpy as np
from ..utils import normalize,polynomial_features
from ..utils import batch_generator

class l1_regularization():
    """
    Regularization for Lasso Regression
    """
    def __init__(self,alpha):
        self.alpha = alpha

    def __call__(self,w):
        return self.alpha*np.linalg.norm(w,ord=1)

    def grad(self,w):
        return self.alpha * np.sign(w)

class l2_regularization():
    """
    Regularization for Ridge Regression
    """
    def __init__(self,alpha):
        self.alpha = alpha

    def __call__(self,w):
        return self.alpha * 0.5 * w.T.dot(w)

    def grad(self,w):
        return self.alpha * w

class l1_l2_regularization():
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
    Base regression class using stochastic gradient descent.
    This class should not be called by users.

    The formula is based on this:
    https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation

    Parameters
    ------------------------------
    max_iter: float
        The maximum number of training iterations
    learning_rate: float
        The step length that will be used when updating the weights
    """
    def __init__(self,
                 max_iter,
                 learning_rate,
                 coef_init=None,
                 intercept_init=None,
                 tol=1e-3):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.coef_init = coef_init
        self.intercept_init = intercept_init
        self.tol = tol
        self.n_iter=0

    def initialize_weights(self,n_features):
        """
        Initialize weights randomly [-1/N,-1/N]
        """
        limit=1/np.sqrt(n_features)
        self.w = np.random.uniform(-limit,limit,(n_features,))
        self.bias = np.zeros((1,))

    def fit(self,X,y,batch_size=-1):
        # Insert constant ones for bias weights as the last column
        if self.coef_init is not None and self.intercept_init is not None:
            self.w = self.coef_init
            self.bias = self.intercept_init
        else:
            self.initialize_weights(n_features=X.shape[1])

        # batch_size = -1 means Gradient Descent
        if batch_size == -1:
            batch_size=X.shape[0]

        best_loss = np.inf
        n_samples = X.shape[0]

        # Do gradient descent for n_iterations
        for i in range(self.max_iter):
            this_loss = 0
            for X_batch,y_batch in batch_generator(X,y,batch_size = batch_size):
                this_batch_size = X_batch.shape[0]
                batch_y_pred = X_batch@self.w+self.bias
                # calculate l2 loss
                mse = np.sum(0.5*(y_batch-batch_y_pred)**2)
                this_loss+=mse
                # Gradient of l2 loss w.r.t w
                dloss = X_batch.T@(-(y_batch-batch_y_pred))/this_batch_size
                if self.penalty_type == 'l2':
                    dr = self.regularization.grad(self.w)
                self.dw = dloss + dr
                self.db = np.sum(-(y_batch-batch_y_pred))
                # update weights
                self.w -= self.learning_rate*self.dw
                self.bias -= self.learning_rate*self.db
            if best_loss - this_loss<self.tol*n_samples:
                self.n_iter=i+1
                break
            if this_loss<best_loss:
                best_loss=this_loss
        self.n_iter=i+1

    def predict(self,X):
        X = np.insert(X,0,1,axis=1)
        y_pred = X.dot(self.w)
        return y_pred


# least square method matrix form
# https://math.stackexchange.com/questions/369694/matrix-calculus-in-least-square-method
# formula:
# https://math.stackexchange.com/questions/644834/least-squares-in-a-matrix-form
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
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse just in case)
            # the matrix is not invertable
            U,S,V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S).dot(U.T))
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression,self).fit(X,y)

    def predict(self,X):
        # Insert constant ones
        X = np.insert(X,0,1,axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class PolynominalRidgeRegression(Regression):
    """
    Linear regression model with a l2 regularization factor

    Parameters
    ----------------------------------------
    degree:int
        The degree of the polynominal that the independent variable X will be
        transformed to.
    reg_factor: float
        The factor that determines the degree of regularization and feature shrinkage
    n_iterations: float
        The number of training iterations
    learning_rate: float
    """
    def __init__(self,degree,reg_factor,n_iterations=3000,learning_rate=0.01):
        self.degree=degree
        self.regularization=l2_regularization(alpha=reg_factor)
        super(PolynominalRidgeRegression,self).__init__(n_iterations,learning_rate)

    def fit(self,X,y):
        X = normalize(polynomial_features(X,degree=self.degree))
        super(PolynominalRidgeRegression,self).fit(X,y)

    def predict(self,X):
        X = normalize(polynomial_features(X,degree=self.degree))
        return super(PolynominalRidgeRegression,self).predict(X)


class RidgeRegression(Regression):
    """
    Parameters
    -----------------------
    alpha: float
        The factor that determines the degree of regularization
    max_iter: float
        The maximum number of training iterations
    learning_rate: float
        The step length that will be used
    """
    def __init__(self,
                 alpha,
                 max_iter,
                 learning_rate,
                 coef_init,
                 intercept_init,
                 tol):
        self.regularization = l2_regularization(alpha=alpha)
        self.penalty_type='l2'
        super(RidgeRegression,self).__init__(max_iter,
                                             learning_rate,
                                             coef_init,
                                             intercept_init,
                                             tol)

class LassoRegression(Regression):
    """
    Parameters
    -----------------------
    alpha: float
        The factor that determines the degree of regularization
    max_iter: float
        The maximum number of training iterations
    learning_rate: float
        The step length that will be used
    """
    def __init__(self,
                 alpha,
                 max_iter,
                 learning_rate,
                 coef_init,
                 intercept_init,
                 tol):
        self.regularization = l1_regularization(alpha=alpha)
        super(LassoRegression,self).__init__(max_iter,
                                             learning_rate,
                                             coef_init,
                                             intercept_init,
                                             tol)

class ElasticNet(Regression):
    """
    Regression with the combination of l1 and l2 regularization.

    Parameters
    -------------------------------
    reg_factor: float
        The degree of regularization
    l1_ratio: float
        The degree of l1 regularization at the combination of l1 and l2
    regularization
    n_iterations: n
        The number of training iterations
    learning_rate:float
        The step length
    """
    def __init__(self,reg_factor=0.05,l1_ratio=0.5,n_iterations=100,learning_rate=0.01):
        self.regularization = l1_l2_regularization(alpha=reg_factor,l1_ratio=l1_ratio)
        super(ElasticNet,self).__init__(n_iterations=n_iterations,learning_rate=learning_rate)
