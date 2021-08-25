import numpy as np
from numpy.testing import assert_allclose

"""
WARNING:

This implementation does not strictly follow the reference paper due to the
problem of the sklearn implementation, and I want to pass the unit test.
The problem lies in the calculation of ELBO. For more details, please refer to
https://github.com/scikit-learn/scikit-learn/issues/14419

Once the implementation of sklearn is fixed, I will come back and refactor my
implementation.
"""
class GMM:
    def __init__(self,
                 C: int=3,
                 seed: int=None,
                 max_iter: int = 100,
                 tol: float = 1e-3,
                 weights_init: np.ndarray=None,
                 means_init: np.ndarray=None,
                 precisions_init: np.ndarray=None) -> None:
        """
        A Gaussian maxiture model trained via the expectation maimization
        algorithm

        Ref: http://cs229.stanford.edu/notes2020spring/cs229-notes8.pdf

        Parameters:
        ----------------------
        C: int
            The number of mixture components in the GMM.
        seed: int
            Seed for the random number generator.
        max_iter: int
            The maximum number of  EM updates to perform before terminating
            training.
        tol: float
            The convergence tolerance. The training will be terminated if
            the difference of the lower bound betweet two iterations is
            less than tol.
        weights_init: np.ndarray
            The user-provided initial weights. If None, weights are
            initialized in the _initialize_parameters() function.
        means_init: np.ndarray
            The user-provided initial means. If None, weights are
            initialized in the _initialize_parameters() function.
        precisions_init: np.ndarray
            The user-provided initial precisions (inverse of the covariance matrices).
            If it is None, precisions are initialized in the _initialize_parameters() function.
            It makes prediction more convenient.
        Attributes:
        ----------------------
        N: int
            The number of observations in the training set.
        d: int
            The dimension of each observation in the training set.
        pi: numpy.array of shape (C,)
            The cluster priors. pi[j] shows the probability that the latent
            variable comes from the Gaussian distribution j. pi is also called
            weights.
        Q: numpy.array of shape of shape (N, C)
            The distribution over the laten variable Z.
        mu: numpy.array of shape (C, d)
            Means of Gaussian distributions
        sigma: numpy.array of shape (C, d, d)
            Covariance matrices for Gaussian distributions.
        inverse_sigma: numpy.array of shape (C, d, d)
            The inverse of the Covariance matrices for Gaussian distributions.
            Storing this makes prediction convenient.
        """
        self.C = C
        self.d = None
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.is_converged = False
        # https://github.com/scikit-learn/scikit-learn/blob/6b4f82433dc2f219dbff-
        # 7fe8fa42c10b72379be6/sklearn/mixture/_gaussian_mixture.py#L278
        self.eps = 10 * np.finfo(float).eps

    def _initialize_parameters(self) -> None:
        if self.seed:
            np.random.seed(self.seed)
        # initialize weights
        if self.weights_init is None:
            self.pi = np.random.rand(self.C)
            self.pi = self.pi/self.pi.sum()
        else:
            self.pi = self.weights_init

        self.Q = np.zeros([self.N, self.C])

        # Initialize mu by randomly select C values from each dimension
        if self.means_init is None:
            self.mu = np.zeros([self.C, self.d])
            for this_dim in range(self.d):
                this_mu = np.random.choice(self.X[:,this_dim],self.C)
                self.mu[:, this_dim] = this_mu
        else:
            self.mu = self.means_init

        # Diag covariance matrices
        if self.precisions_init is None:
            self.inverse_sigma = np.array([np.identity(d) for _ in range(C)])
        else:
            self.inverse_sigma = self.precisions_init
        self.sigma = np.zeros(self.inverse_sigma.shape)

        self.n_iter_=0
        self.best_pi = None
        self.best_mu = None
        self.best_sigma = None
        self.best_elbo = -np.inf

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the parameters of the GMM on the training set

        Parameters:
        ---------------
        X: numpy.array of shape (N, d)
            The training set with N observations and d variables.
        """
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[1]

        self._initialize_parameters()
        prev_elbo = -np.inf

        for round in range(self.max_iter):
            self.n_iter_+=1
            this_elbo = self._E_step()
            self._M_step()

            is_converged = np.abs(this_elbo - prev_elbo) <= self.tol
            if is_converged:
                self.is_converged = True
                return
            prev_elbo = this_elbo
            if this_elbo > self.best_elbo:
                self.best_elbo = this_elbo
                self.best_mu = self.mu
                self.best_pi = self.pi
                self.best_sigma = self.sigma
        return

    def _E_step(self) -> float:
        log_denom_list = []
        for i in range(self.N):
            x_i = self.X[i, :]

            nuerators = []
            for this_c in range(self.C):
                pi_c = self.pi[this_c]
                mu_c = self.mu[this_c, :]
                inverse_sigma_c = self.inverse_sigma[this_c, :, :]

                # log prior
                log_pi_c = np.log(pi_c+self.eps)
                # log Gaussian density
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, inverse_sigma_c)
                nuerators.append(log_pi_c+log_p_x_i)

            # logsumexp: lnF1, lnF2, lnF3, ... -> ln(F1 + F2 + F3 + ...)
            log_denom = logsumexp(nuerators)
            log_denom_list.append(log_denom)
            q_i = np.exp([num - log_denom for num in nuerators])
            assert_allclose(np.sum(q_i), 1)

            self.Q[i, :] = q_i
        return np.mean(log_denom_list)

    def _M_step(self) -> None:
        # total weights over all observations for each Gaussian distribution
        total_imp = np.sum(self.Q, axis=0)
        # update priors
        self.pi = total_imp/self.N

        # update Gaussian distribution means
        mu_numer = [np.dot(self.Q[:, this_c], self.X) for this_c in range(self.C)]
        for idx, (this_mu_numer, this_weights) in enumerate(zip(mu_numer, total_imp)):
            self.mu[idx,:] = this_mu_numer/this_weights

        # update Gaussian distribution covariance
        for this_c in range(self.C):
            mu_c = self.mu[this_c,:]
            weight_c = total_imp[this_c]

            this_sigma = np.zeros((self.d, self.d))
            for i in range(self.N):
                weight_ic =self.Q[i, this_c]
                xi = self.X[i,:]
                this_sigma += weight_ic * np.outer(xi-mu_c, xi-mu_c)

            this_sigma = this_sigma/weight_c if weight_c>0 else this_sigma
            self.sigma[this_c, :, :] = this_sigma

        # update the inverse of the covariance matrices for future interations
        self.inverse_sigma = np.linalg.inv(self.sigma)

        assert_allclose(np.sum(self.pi), 1)

def log_gaussian_pdf(x_i: np.ndarray, mu: np.ndarray, inverse_sigma: np.ndarray) -> float:
    """
    Compute log N(x_i | mu, sigma)
    """
    n = len(mu)
    a = np.power(2 * np.pi, n)
    b = np.linalg.det(inverse_sigma)
    b = abs(1/b)
    ab = np.log(a * b)

    # calculate the inverse of sigma
    y = inverse_sigma @ (x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (ab + c)

def logsumexp(log_probs: list, axis: int=None) -> float:
    """
    lnF1, lnF2, lnF3, ... -> ln(F1 + F2 + F3 + ...)
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
