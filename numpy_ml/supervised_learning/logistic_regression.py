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
    Logistic regression classifier with L-BFGS optimization method. No regularization
    is applied.

    The reference for the L-BFGS method:
    Representations of quasi-Newton matrices and their use in limited memory methods
    https://www.semanticscholar.org/paper/Representations-of-quasi-Newton-matrices-and-their-Byrd-Nocedal/dff7bb898da45b502608c3603b4673315540d4fd
    """
    def __init__(self,
                 max_iter: int=100,
                 tol: float=1e-3,
                 maxcor: int=10,
                 ftol: float=2.2204460492503131e-09,
                 gtol: float=1e-5) -> None:
        """
        max_iter: int
            The maximum number of iterations
        tol: float
            The tolerance for stopping criteria
        maxcor: int
            The maximum number of variable metric corrections used to
            define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms
            in an approximation to it.)
        ftol: float
            mu in the backtracking line search reference paper. It controls the
            minimum decrease of the function value.
        gtol: float
            eta in the backtracking line search reference paper. It controls the
            minimum decrease of the gradient.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.maxcor = maxcor
        self.ftol = ftol
        self.gtol = gtol


    def _initialize_parameters(self, X: np.ndarray, init_w: bool = True, init_b: bool=True) -> None:
        _, n_feat = np.shape(X)
        # formula: x*w+b
        if init_w:
            self.w = np.zeros((n_feat,))
        if init_b:
            self.b = np.zeros((1,))

    def _loss_and_grad(self, y_true, z):
        """
        This is the loss function for logistic regression when the label is +1 and
        -1. The formula is log(1+exp(-y*z)), where z = xw+b. I do not put it into
        a separate file because it is only used here.

        Reference:
        https://stats.stackexchange.com/questions/250937/which-loss-function-is-correct-for-logistic-regression/279698#279698?newreg=78c6493a7c9e49e381a74845b7a4ddb0
        """
        # calculate loss
        yz = y_true*z
        result = np.zeros_like(y_true)
        for i in range(yz):
            this_yz = yz[i]
            if this_yz > 0:
                result[i] = np.log(1+np.exp(-this_yz))
            else:
                result[i] = -this_yz + np.log(1+np.exp(this_yz))

        # calculate grad
        z = 1/(1+np.exp(-yz))
        dz = (z - 1)*y_true
        dw = self.X.T@dz
        # dw[-1] is actually the gradient of the bias term.
        dw[-1] = dz.sum()
        return result, dw

    def _backtracking_line_search(self,
                                  f: float,
                                  direction: np.ndarray,
                                  g: np.array,
                                  stpmin: float = 0,
                                  stpmax :float=1e10,
                                  mu :float = self.ftol,
                                  eta: float = self.gtol) -> np.ndarray:
        """
        Perform the Wolfe line search method to get the best step length.

        Reference:
        On line search algoithms with guaranteed sufficient decrease
        https://www.researchgate.net/publication/220493298_On_Line_Search_Algorithms_with_Guaranteed_Sufficient_Decrease

        f: float
            The evaluation of the function to minimize
        direction: np.array
            The direction of the optimization method. For the first iterate, the
            direction is equal to the first order gradient w.r.t. the input x
        g: np.array
            The gradient of the function w.r.t. the input x
        stpmax: float
            The upper bound of the step length. The default value comes from
            sklearn.
            https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb_src/lbfgsb.f#L2485
        sptmin: float
            The lower bound of the step length.
        """
        dnorm = np.linalg.norm(direction)
        # the initial value of the best step length
        stp = min(1/dnorm, stpmx)
        bracketed = False
        # gd is the gradient w.r.t. the step length (not the input x)
        gd = np.dot(direction, g)
        finit = f
        ginit = gd
        stage = 1
        gtest = mu * ginit
        # lower bound of the step length
        stx = 0
        fx = finit
        gx = ginit
        # upper bound of the step length
        sty = 0
        fy = finit
        gy = ginit
        stmin = 0
        stmax = 5 * stp
        stage = 1

        while True:
            ftest = finit + stp * gtest
            if f <= ftest and abs(direction) <= eta * (-ginit):
                return
            if f <= fx and and f >= ftest:
                fm = f - stp * gtest
                fxm = fx - stx * gtest
                fym = fy - sty * gtest
                gm = g - gtest
                gxm = gx - gtest
                gym = gy - gtest

                self._linesearch_helper(stx, fxm, gxm, sty, fym, gym,
                                        stp, fm, gm, bracketed, stmin, stmax)

                fx = fxm + stx * gtest
                fy = fym + sty * gtest
                gx = gxm + gtest
                gy = gym + gtest
            else:
                self._linesearch_helper(stx, fx, gx, sty, fy, gy, stp, f, g,
                                        bracketed, stmin, stmax)

    def _linesearch_helper(self,
                           stx: float,
                           fx: float,
                           dx: float,
                           sty: float,
                           fy: float,
                           dy: float,
                           stp: float,
                           fp: float,
                           dp: float,
                           bracketed: bool,
                           stpmin: float,
                           stpmax: float) -> ?:
        """
        This helper function returns the best step and updates an interval
        that contains the step that statisfies a sufficient decrease and a
        curvature condition.

        Parameters:
        --------------------
        stx: float
            stx is the endpoint of the interval that contains the best step.
        fx: float
            fx is the function value at stx
        dx: float
            dx is the first order gradient w.r.t. the step length stx (please be
            be aware that dx is not a np.ndarray).
        sty: float
            sty is the second endpoint of the interval that contains the best step
        fy: float
            fy is the function value at sty
        dy: float
            dy is the first order gradient w.r.t. the step length sty (please be
            be aware that dx is not a np.ndarray).
        stp: float
            stp is the current step
        fp: float
            fp is the current function value at stp
        dp: float
            dp is the first order gradient w.r.t. the step length stp (please be
            be aware that dx is not a np.ndarray).
        bracketed: bool
            It shows whether the interval has been bracketed.
        stpmin: float
            A lower bound for the step
        stpmax: float
            An upper bound for the step.
        """
        sgnd = dp * (dx/abs(dx))
        if fp > fx:
            theta = 3*(fx-fp)/(stp-stx) + dx + dp
            s = max(abs(theta), abs(dx), abs*(dp))



    def _l_bfgs(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> :
        S = np.array([np.nan]*self.maxcor)
        Y = np.array([np.nan]*self.maxcor)

        z = X@weights
        fx, grad = self._loss_and_grad(y, z)
        # check whether it is the first iterate
        if np.isnan(S).all():
            # if it is the first iterate, we use the gradient as the direction
            # Reference: http://www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf
            # page 15
            weights_next = self._backtracking_line_search(fx, -grad, grad)



    def _param_check(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
        n_obs, n_feat = np.shape(X)
        assert isinstance(X, np.ndarray), 'The type of X is not understood'
        assert isinstance(y, np.ndarray), 'The type of y is not understood'
        assert isinstance(w, np.ndarray), 'The type of w is not understood'
        assert isinstance(b, np.ndarray), 'The type of b is not understood'
        assert len(y) == n_obs, "The length of X and y should be equal"
        assert len(np.unique(y)) == 2, "The unique length of the target variable should be 2 (binary classification)"
        assert n_feat == w.shape[0], "The shape of the training set and weights does not match"
        assert b.shape[0] == 1, "The shape of the bias term is not correct"

    def fit(self, X: np.ndarray, y: np.ndarray, w_init: np.ndarray=None, b_init: np.ndarray=None) -> None:
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
        # initialize parameters
        if self.w_init is None and self.b_init is None:
            self._initialize_parameters(X)
        elif self.w_init is None:
            self._initialize_parameters(X, init_w=True, init_b=False)
            self.b = b_init
        elif self.b_init is None:
            self._initialize_parameters(X, init_w=False, init_b=True)
            self.w = w_init
        self._param_check(X, y, self.w, self.b)
        # I force the label to be -1 and +1 for the learning and unit test purpose
        if np.unique(y) == np.array([0,1]):
            y[y==0] = -1

        # for the sake of convenience and unit test, we combine weights and the bias term
        n_obs, n_var = np.shape(X)
        self.wb = np.concatenate([self.w,self.b])
        extra_col = np.ones((n_obs,1))
        self.X = np.append(X, extra_col, axis=1)
