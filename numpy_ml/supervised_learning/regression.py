import numpy as np
from ..utils import normalize,polynomial_features
from ..utils import batch_generator

# the implementation of l1 regularization has been removed because its
# calculation is tightly associated with the loss, so I think it is better
# to combine it with the function class

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

    The formula for both l1 and l2 regularization is based on this:
    https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation

    Please be aware that the optimization method used for l1 regularization is
    not the vanilla proximal gradient descent method. Instead, it comes from
    https://www.aclweb.org/anthology/P09-1054.pdf

    Parameters
    ------------------------------
    max_iter: float
        The maximum number of training iterations
    learning_rate: float
        The step length that will be used when updating the weights
    coef_init: np.array of shape (n_features,)
        The initialized weights term
    intercept_init: np.array of shape (1,)
        The initialized bias term
    tol: float
        The minimum required difference of the squared loss between two iterations.
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
        if self.penalty_type == 'l1':
            q = np.zeros(self.w.shape)
            u = 0

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
                    self.dw = (dloss + self.regularization.grad(self.w))
                    self.w -= self.learning_rate*self.dw
                elif self.penalty_type == 'l1':
                    if self.method == 'pgd':
                        u += self.learning_rate*self.alpha
                        for idx, this_dloss in enumerate(dloss):
                            this_regular_w = self.w[idx]-self.learning_rate*this_dloss
                            if this_regular_w > 0:
                                self.w[idx]=max(0, this_regular_w-u-q[idx])
                            elif this_regular_w < 0:
                                self.w[idx]=min(0, this_regular_w+u-q[idx])
                            else:
                                self.w[idx]=0
                            q[idx] += self.w[idx]-this_regular_w
                self.db = np.sum(-(y_batch-batch_y_pred))
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
    coef_init: np.ndarray of shape (n_features,1)
        The initial weights
    intercept_init: np.ndarray of shape (1,)
        The initial bias term
    tol: float
        The tolerance for the optimization
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
    Perform lasso linear regression with proximal gradient descent

    Parameters
    -----------------------
    alpha: float
        The factor that determines the degree of regularization
    max_iter: int
        The maximum number of training iterations
    learning_rate: float
        The step length that will be used
    coef_init: np.ndarray of shape (n_features,1)
        The initial weights
    intercept_init: np.ndarray of shape (1,)
        The initial bias term
    tol: float
        The tolerance for the optimization
    """
    def __init__(self,
                 alpha,
                 max_iter,
                 learning_rate,
                 coef_init,
                 intercept_init,
                 tol):

        self.alpha=alpha
        self.penalty_type = 'l1'
        self.method = "pgd"
        super(LassoRegression,self).__init__(max_iter,
                                             learning_rate,
                                             coef_init,
                                             intercept_init,
                                             tol)

class LassoRegressionCD:
    """
    Perform lasso linear regression with coordinate descent method. I decided not
    to inherit from the regression class because both the optimization and the
    stop criterion are totally different from that of the regression class.

    Reference:
    Page 13 of
    http://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/coord-desc.pdf

    Parameters
    -----------------
    alpha: float
        The factor that determines the degree of regularization
    max_iter: int
        The maximum number of training iterations
    coef_init: np.ndarray of shape (n_features,1)
        The initial weights
    intercept_init: np.ndarray of shape (1,)
        The initial bias term
    tol: float
        The tolerance for the optimization
    """
    def __init__(self,
                 alpha: float,
                 max_iter: int=100,
                 coef_init: np.ndarray=None,
                 intercept_init: np.ndarray=None,
                 tol: float=1e-3) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_init = coef_init
        self.intercept_init = intercept_init
        self.tol = tol
        self.this_iter = 0
        self.dual_gap = 0
        # coordinate descent method has no learning rate

    def _param_initialization(self) -> None:
        if self.coef_init is not None and self.intercept_init is not None:
            self.w  = self.coef_init
            self.b = self.intercept_init
        elif self.coef_init is not None:
            self.w  = self.coef_init
            self.b = np.zeros((1,))
        elif self.intercept_init is not None:
            self.w = np.zeros((self.n_features,))
            self.b = self.intercept_
        else:
            self.w = np.zeros((self.n_features,))
            self.b = np.zeros((1,))

    def fit(self, X, y) -> None:
        n_obs, self.n_features = X.shape
        self._param_initialization()
        y_offset = np.average(y, axis=0)
        X_offset = np.average(X, axis=0)
        X -= X_offset
        y -= y_offset
        self.alpha *= n_obs
        residual = y - X@self.w
        for _ in range(self.max_iter):
            max_w = 0
            max_diff = 0
            self.this_iter+=1
            for i in range(self.n_features):
                denominator = X[:,i].T@X[:,i]
                if denominator == 0:
                    continue
                w_i = self.w[i]
                if w_i != 0:
                    # Current residual is the residual except the current feature
                    residual += X[:,i]*w_i
                numerator = X[:,i].T@residual
                self.w[i] = np.sign(numerator) * max(abs(numerator) - self.alpha, 0)\
                                 / (denominator)
                if self.w[i] != 0:
                    # Current residual includes all the features
                    residual -= X[:,i]*self.w[i]
                this_diff = abs(self.w[i]-w_i)
                max_diff = max(max_diff, this_diff)
                max_w = max(max_w, abs(self.w[i]))

            # if the update is small or reaches the max iter, calculate the
            # duality gap
            if max_w == 0 or max_diff/max_w < self.tol or self.this_iter == self.max_iter:
                Xu = X.T@residual
                Xu_inf_norm = np.linalg.norm(Xu,np.inf)
                residual_2norm = residual.T@residual
                if Xu_inf_norm > self.alpha:
                    const = self.alpha/Xu_inf_norm
                    self.dual_gap = 0.5*(residual_2norm+residual_2norm*(const**2))
                else:
                    const = 1
                    self.dual_gap = residual_2norm
                self.dual_gap += (self.alpha*np.linalg.norm(self.w,1)-const*residual.T@y)

                if self.dual_gap < self.tol*np.dot(y,y):
                    break

        # self.w = wb[:-1]
        self.b = y_offset - X_offset@self.w



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
