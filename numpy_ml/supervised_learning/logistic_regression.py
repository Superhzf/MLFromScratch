import numpy as np
import math
from ..deep_learning.activation_functions import Sigmoid

# Q: Diagnose logistic regression
# A: https://stats.idre.ucla.edu/stata/webbooks/logistic/chapter3/lesson-3-logistic-regression-diagnostics/
# https://www.r-bloggers.com/evaluating-logistic-regression-models/
class LogisticRegression():
    """
    Logistic regression Classifier
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


class LogisticRegression_LBFGS:
    """
    Logistic regression classifier with L-BFGS optimization method.

    The reference for deriving the loss function:
    https://stats.stackexchange.com/questions/250937/which-loss-function-is-correct-for-logistic-regression/279698#279698?newreg=78c6493a7c9e49e381a74845b7a4ddb0

    The reference for the L-BFGS method:
    Representations of quasi-Newton matrices and their use in limited memory methods
    https://www.semanticscholar.org/paper/Representations-of-quasi-Newton-matrices-and-their-Byrd-Nocedal/dff7bb898da45b502608c3603b4673315540d4fd

    The reference for the backtracking line search method:
    On line search algoithmswith guaranteed sufficient decrease
    https://www.researchgate.net/publication/220493298_On_Line_Search_Algorithms_with_Guaranteed_Sufficient_Decrease

    """
    def __init__(self, max_iter: int=100, tol: float=1e-3) -> None:
        """
        max_iter: int
            The maximum number of iterations
        tol: float
            The tolerance for stopping criteria
        """
        self.max_iter = max_iter
        self.tol = tol

    def _initialize_parameters(self, X: np.array, init_w: bool = True, init_b: bool=True) -> None:
        _, n_feat = np.shape(X)
        # formula: x*w+b
        if init_w:
            self.w = np.zeros((n_feat,))
        if init_b:
            self.b = np.zeros((1,))

    def _param_check(self, X: np.array, y: np.array, w: np.array, b: np.array) -> None:
        n_obs, n_feat = np.shape(X)
        assert isinstance(X, np.ndarray), 'The type of X is not understood'
        assert isinstance(y, np.ndarray), 'The type of y is not understood'
        assert isinstance(w, np.ndarray), 'The type of w is not understood'
        assert isinstance(b, np.ndarray), 'The type of b is not understood'
        assert len(y) == n_obs, "The length of X and y should be equal"
        assert n_feat == w.shape[0], "The shape of the training set and weights does not match"
        assert b.shape[0] == 1, "The shape of the bias term is not correct"

    def fit(self, X: np.array, y: np.array, w_init: np.array=None, b_init: np.array=None) -> None:
        """
        Fit the logistic regression with L_BFGS method

        Parameters:
        -------------------
        X: np.array of shape (n_obs, n_feat)
            The training set, where n_obs is the number of observations and n_feat
            is the number of variables.
        y: np.array of shape (n_obs,)
            The target vector, where n_obs is the number of observations.
        w_init: np.array of shape (n_feat,)
            The initialized weights, where n_feat is the number of variables.
        b_init: np.array of shape (1,)
            The initialized bias term.
        """
        if self.w_init is None and self.b_init is None:
            self._initialize_parameters(X)
        elif self.w_init is None:
            self._initialize_parameters(X, init_w=True, init_b=False)
            self.b = b_init
        elif self.b_init is None:
            self._initialize_parameters(X, init_w=False, init_b=True)
            self.w = w_init
        self._param_check(X, y, self.w, self.b)
        
