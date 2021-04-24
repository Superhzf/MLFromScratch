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
                 maxcor: int=10,
                 ftol: float=1e-3,
                 gtol: float=0.9) -> None:
        """
        max_iter: int
            The maximum number of iterations
        maxcor: int
            The maximum number of variable metric corrections used to
            define the limited memory matrix. (The limited memory BFGS
            method does not store the full hessian but uses this many terms
            in an approximation to it.)
        ftol: float
            mu in the backtracking line search reference paper. It controls the
            minimum decrease of the function value. The value comes from sklearn.
        gtol: float
            eta in the backtracking line search reference paper. It controls the
            minimum decrease of the gradient.
        """
        self.max_iter = max_iter
        self.this_iter = 0
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
        result = 0
        for i in range(len(yz)):
            this_yz = yz[i]
            if this_yz > 0:
                result += np.log(1+np.exp(-this_yz))
            else:
                result += -this_yz + np.log(1+np.exp(this_yz))

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
                                  mu: float,
                                  eta: float,
                                  stpmin: float = 0,
                                  stpmax: float = 1e10,
                                  xtol: float = 0.1) -> float:
        """
        Perform the Wolfe line search method to get the best step length.

        Reference:
        On line search algoithms with guaranteed sufficient decrease
        https://www.researchgate.net/publication/220493298_On_Line_Search_Algorithms_with_Guaranteed_Sufficient_Decrease

        Parameters:
        ---------------------
        f: float
            The evaluation of the function to minimize
        direction: np.array
            The direction of the optimization method. For the first iterate, the
            direction is equal to the first order gradient w.r.t. the input x.
        g: np.array
            The gradient of the function w.r.t. the input x. g and direction to
            gether are used to calculate the gradient w.r.t. the step length
        sptmin: float
            The lower bound of the step length.
        stpmax: float
            The upper bound of the step length. The default value comes from
            sklearn.
            https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb_src/lbfgsb.f#L2485
        mu: float
            The factor used in line search to spin the line.
        eta: float
            The factor used in line search on the gradient at the initial point,
            the new gradient should be less than the gradient by eta.
        xtol: float
            The relative difference between sty and stx should be larger than xtol.
        """
        # if it is the first iterate
        if self.this_iter == 1:
            dnorm = np.linalg.norm(direction)
            # the initial value of the best step length
            stp = min(1/dnorm, stpmax)
        else:
            stp = 1
        bracketed = False
        # gd is the gradient of the func w.r.t. the step length (not the input x)
        gd = np.dot(direction, g)
        finit = f
        ginit = gd
        stage = 1
        gtest = mu * ginit
        width = (stpmax-stpmin)
        # Per the paper, the bisection setp is used when after two trials, the
        # length decrease does not meet the factor 0.66. But sklearn just
        # multiplies 2. I do the same thing for the unit test purpose.
        width2 = 2*(stpmax-stpmin)
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

        this_wb = self.wb + stp*direction
        f, g = self._loss_and_grad(self.y, self.X@this_wb)
        gd = np.dot(direction, g)

        while True:
            ftest = finit + stp * gtest
            if (stage == 1 and f <= ftest and gd >= 0):
                stage = 2
            # return if converged
            if f <= ftest and abs(gd) <= eta * (-ginit):
                print ('1111')
                return stp
            # return if stp is out of the range
            if (bracketed and (stp <= stmin or stp >= stmax)):
                print ('2222')
                return stp
            # return if the length of the interval is too short
            if (bracketed and stmax - stmin <= xtol*stmax):
                print ('3333')
                return stp
            # return if stp == stpmax
            if (stp == stpmax and f <= ftest and gd <= gtest):
                print ('4444')
                return stp
            # return if stp == stpmin
            if (stp == stpmin and (f > ftest or gd >= gtest)):
                print ('5555')
                return stp

            if stage == 1 and f <= fx and f > ftest:
                fm = f - stp * gtest
                fxm = fx - stx * gtest
                fym = fy - sty * gtest
                gm = gd - gtest
                gxm = gx - gtest
                gym = gy - gtest

                stp, stx, fx, dx, sty, fy, dy, bracketed = self._linesearch_helper(stx,
                                                     fxm, gxm, sty, fym, gym,
                                         stp, fm, gm, bracketed, stmin, stmax)

                fx = fxm + stx * gtest
                fy = fym + sty * gtest
                gx = gxm + gtest
                gy = gym + gtest
            else:
                stp, stx, fx, dx, sty, fy, dy, bracketed = self._linesearch_helper(stx,
                                                         fx, gx, sty, fy, gy,
                                          stp, f, gd, bracketed, stmin, stmax)
            if bracketed:
                # the length of the interval does not decrease by 0.66 (0.66
                # comes from the paper without mathematical reasons)
                if abs(sty-stx) >= 0.66*width2:
                    stp = stx + 0.5*(sty - stx)
                    width2 = width
                    width = abs(sty-stx)

            if bracketed:
                stmin = min(stx,sty)
                stmax = max(stx,sty)
            else:
                stmin = stp + 1.1*(stp - stx)
                stmax = stp + 4*(stp - stx)

            stp = max(stp,stpmin)
            stp = min(stp,stpmax)

            if ((bracketed and (stp <= stmin or stp >= stmax)) or \
                (bracketed and (stmax-stmin) <= xtol*stmax)):
                stp = stx

            this_wb = self.wb + stp*direction
            f, g = self._loss_and_grad(self.y, self.X@this_wb)
            gd = np.dot(direction, g)

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
                           stpmax: float) -> float:
        """
        This helper function performs one iterate of line search and returns the
        best step and updates an interval that contains the step that statisfies
        a sufficient decrease and a curvature condition.

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
        dpx = dp*dx
        if fp > fx:
            theta = 3*(fx-fp)/(stp-stx) + dx + dp
            s = max(abs(theta), abs(dx), abs(dp))
            gamma = s*np.sqrt((theta/s)**2 - (dx/s)*(dp/s))
            if (stp < stx):
                gamma = -gamma
            p = (gamma - dx) + theta
            q = ((gamma - dx) + gamma) + dp
            r = p/q
            stpc = stx + r*(stp - stx)
            stpq = stx + ((dx/((fx - fp)/(stp - stx) + dx))/2)*(stp - stx)
            if (abs(stpc-stx) < abs(stpq-stx)):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc)/2
            bracketed = True
        elif dpx<0:
            theta = 3*(fx - fp)/(stp - stx) + dx + dp
            s = max(abs(theta), abs(dx), abs(dp))
            gamma = s*np.sqrt((theta/s)**2 - (dx/s)*(dp/s))
            if (stp > stx):
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dx
            r = p/q
            stpc = stp + r*(stx - stp)
            stpq = stp + (dp/(dp - dx))*(stx - stp)
            if (abs(stpc-stp) > abs(stpq-stp)):
                stpf = stpc
            else:
                stpf = stpq
            bracketed = True
        elif abs(dp) < abs(dx):
            theta = 3*(fx - fp)/(stp - stx) + dx + dp
            s = max(abs(theta), abs(dx), abs(dp))
            gamma = s*np.sqrt(max(0,(theta/s)**2-(dx/s)*(dp/s)))
            if (stp > stx):
                gamma = -gamma
            p = (gamma - dp) + theta
            q = (gamma + (dx - dp)) + gamma
            r = p/q
            if (r < 0 and gamma != 0):
                stpc = stp + r*(stx - stp)
            elif stp>stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + (dp/(dp - dx))*(stx - stp)
            if bracketed:
                if (abs(stpc-stp) < abs(stpq-stp)):
                    stpf = stpc
                else:
                    stpf = stpq
                if (stp > stx):
                    stpf = min(stp+0.66*(sty-stp),stpf)
                else:
                    stpf = max(stp+0.66*(sty-stp),stpf)
            else:
                if (abs(stpc-stp) > abs(stpq-stp)):
                    stpf = stpc
                else:
                    stpf = stpq
                stpf = min(stpmax,stpf)
                stpf = max(stpmin,stpf)
        else:
            if bracketed:
                theta = 3*(fp - fy)/(sty - stp) + dy + dp
                s = max(abs(theta), abs(dy), abs(dp))
                gamma = s*np.sqrt((theta/s)**2 - (dy/s)*(dp/s))
                if (stp > sty):
                    gamma = -gamma
                p = (gamma - dp) + theta
                q = ((gamma - dp) + gamma) + dy
                r = p/q
                stpc = stp + r*(sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin
        # update the interval
        if fp > fx:
            sty = stp
            fy = fp
            dy = dp
        else:
            if dpx < 0:
                sty = stx
                fy = fx
                dy = dx
            stx = stp
            fx = fp
            dx = dp

        return stpf, stx, fx, dx, sty, fy, dy, bracketed

    def _l_bfgs(self) -> None:
        # we start from the length of 1 and fill it till the length becomes maxcor
        S = np.array([np.nan])
        Y = np.array([np.nan])
        R = np.zeros((1, 1))
        D = np.zeros((1, 1))
        # progress indicates where the new sk and gk should be stored in S and Y
        progress = 0
        # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=349
        epsmch=2*1.110223024625157e-016
        tol = (1e7*np.finfo(float).eps)*epsmch

        z = self.X@self.wb
        fx, grad = self._loss_and_grad(self.y, z)
        # termimate the algorithm if the gradient is too small
        if abs(np.max(grad)) < 1e-5:
            return

        for _ in range(self.max_iter):
            self.this_iter+=1
            # check whether it is the first iterate
            if np.isnan(S).all():
                # if it is the first iterate, we use the gradient as the direction
                # Reference: http://www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf
                # page 15
                step_len = self._backtracking_line_search(fx, -grad, grad,
                                                    mu=self.ftol, eta=self.gtol)
                wb_next = self.wb - step_len * grad
                sk = wb_next - self.wb
                fx_next, grad_next = self._loss_and_grad(self.y, self.X@wb_next)
                if fx - fx_next <= tol*max(abs(fx), abs(fx_next), 1):
                    return
                yk = grad_next - grad
                self.wb = wb_next
                fx = fx_next

            else:
                # update
                S[progress] = sk
                Y[progress] = yk
                sy = sk.T@yk
                if R.shape == (1,1):
                    R[progress, progress] = sy
                elif R.shape[0] < self.maxcor:
                    R = np.pad(R,
                               pad_width=((0, 1), (0, 1)),
                               mode='constant',
                               constant_values=0)
                    _, size_R = R.shape
                    for idx in range(size_R):
                        R[size_R, idx] = S[idx].T@yk
                elif R.shape[0] == self.maxcor:
                    # delete the first  row
                    R = np.delete(R, 0, 0)
                    # delete the first column
                    R = np.delete(R, 0, 1)
                    # add a row and a column at the end
                    R = np.pad(R,
                               pad_width=((0, 1), (0, 1)),
                               mode='constant',
                               constant_values=0)
                    for idx in range(size_R):
                        R[size_R, idx] = S[idx].T@yk
                else:
                    assert R.shape[0] <= self.maxcor,\
                           "The length of R should be less than m"
                R_inv = np.linalg.inv(R)

                D[progress] = sy
                Sg = S.T@grad_next
                Yg = Y.T@grad_next
                yy = Y.T@Y
                grad = grad_next

                gamma = yk.T@sk/yk.T@yk
                p1 = R_inv.T@(D+gamma*yy)@R_inv@Sg-gamma*R_inv.T@Yg
                p2 = -R_inv@Sg
                p = np.concatenate([p1,p2])
                direction = gamma*grad+np.concatenate([S[:,None], gamma*Y[:,None]],axis=1)@p
                if progress+1 < self.maxcor:
                    progress += 1
                else:
                    progress = (progress+1)%self.maxcor

                wb_next = self._backtracking_line_search(fx, -direction, grad,
                                                    mu=self.ftol, eta=self.gtol)
                # wb_next = sel.wb - step_len @ direction
                sk = wb_next - self.wb
                fx_next, grad_next = self._loss_and_grad(self.y, self.X@wb_next)
                if fx - fx_next <= tol*max(abs(fx), abs(fx_next), 1):
                    return
                yk = grad_next - grad
                self.wb = wb_next
                fx = fx_next
                # termimate the algorithm if the gradient is too small
                if abs(np.max(grad_next)) < 1e-5:
                    return

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
        self.X = X
        self.y = y
        # initialize parameters
        if w_init is None and b_init is None:
            self._initialize_parameters(self.X)
        elif w_init is None:
            self._initialize_parameters(self.X, init_w=True, init_b=False)
            self.b = b_init
        elif b_init is None:
            self._initialize_parameters(self.X, init_w=False, init_b=True)
            self.w = w_init
        self._param_check(self.X, self.y, self.w, self.b)
        # I force the label to be -1 and +1 for the learning and unit test purpose
        if (np.unique(self.y) == np.array([0,1])).all():
            self.y[self.y==0] = -1

        # for the sake of convenience and unit test, we combine weights and the bias term
        n_obs, n_var = np.shape(self.X)
        self.wb = np.concatenate([self.w,self.b])
        extra_col = np.ones((n_obs,1))
        self.X = np.append(self.X, extra_col, axis=1)
        self._l_bfgs()
        self.w = self.wb[:-1]
        self.b = np.array([self.wb[-1]])
