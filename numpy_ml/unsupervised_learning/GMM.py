import numpy as np
from numpy.testing import assert_allclose

class GMM(object):
    def __init__(self,
                 C: int=3,
                 seed: int=None,
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
        self.N = N
        self.d = None
        self.seed = seed
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.is_converged = None

        if self.seed:
            np.random.seed(self.seed)

        def _initialize_parameters(self) -> None:
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
                self.sigma = np.array([np.identity(d) for _ in range(C)])
            else:
                self.inverse_sigma = self.precisions_init
                self.sigma = np.zeros(self.inverse_sigma.shape)

            self.best_pi = None
            self.best_mu = None
            self.best_sigma = None
            self.best_elbo = -np.inf

        def likelihood_lower_bound(self) -> float:
            """Calculate the ELBO under the current GMM parameters"""
            eps = np.finfo(float).eps
            elbo = 0.0

            for i in range(sef.N):
                x_i = self.X[i]

                for this_c in range(C):
                    pi_c = self.pi[this_c]
                    z_c = self.Q[i, this_c]
                    mu_c = self.mu[this_c, :]
                    inverse_sigma_c = self.inverse_sigma[this_c, :, :]

                    log_pi_c = np.log(pi_c+eps)
                    log_p_x_i = log_gaussian_pdf(x_i, mu_c, inverse_sigma_c)
                    log_z_c = np.log(z_c+eps)
                    elbo += (z_c * (log_pi_c + log_p_x_i - log_z_c))
            return elbo

        def fit(self, X: np.ndarray, max_iter:int=100, tol:float=1e-3) -> bool:
            """
            Fit the parameters of the GMM on the training set

            Parameters:
            ---------------
            X: numpy.array of shape (N, d)
                The training set with N observations and d variables.
            max_iter: int
                The maximum number of  EM updates to perform before terminating
                training.
            tol: float
                The convergence tolerance. The training will be terminated if
                the difference of the lower bound betweet two iterations is
                less than tol.

            Returns:
            ---------------
            converged: bool
                True: the model is converged.
                False: the model is not converged.
            """
            self.X = X
            self.N = X.shape[0]
            self.d = X.shape[1]

            self._initialize_parameters()
            prev_elbo = -np.inf

            for _ in range(max_iter):
                self._E_step()
                try:
                    self._M_step()
                except np.linalg.LinAlgError:
                    print ('Cannot calculatet the inverse of the covariance matrix')
                    self.is_converged = False
                    return
                this_elbo = self.likelihood_lower_bound()

                is_converged = np.abs(this_elbo - prev_elbo) <= tol
                if is_converged:
                    self.is_converged = True
                    return
                prev_elbo = this_elbo
                if this_elbo > self.best_elbo:
                    self.best_elbo = this_elbo
                    self.best_mu = self.mu
                    self.best_pi = self.pi
                    self.best_sigma = self.sigma
            self.is_converged = True
            return

        def _E_step(self) -> None:
            eps = np.finfo(float).eps
            for i in range(self.N):
                x_i = self.X[i, :]

                denom = []
                for this_c in range(self.C):
                    pi_c = self.pi[this_c]
                    mu_c = self.mu[this_c, :]
                    inverse_sigma_c = self.inverse_sigma[this_c, :, :]

                    log_pi_c = np.log(pi_c+eps)
                    log_p_x_i = log_gaussian_pdf(x_i, mu_c, inverse_sigma_c)
                    denom.append(log_pi_c+log_p_x_i)

            # logsumexp: logF1, logF2, logF3, ... -> log(F1 + F2 + F3 + ...)
            log_denom = logsumexp(denom)
            q_i = np.exp([num - log_denom for num in denom])
            assert_allclose(np.sum(q_i), 1)

            self.Q[i, :] = q_i

        def _M_step(self) -> None:
            # total weights over all observations for each Gaussian distribution
            total_imp = np.sum(self.Q, axis=0)

            # update priors
            self.pi = total_imp/self.N

            # update Gaussian distribution means
            mu_numer = [np.dot(self.Q[:, this_c], X) for this_c in range(C)]
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
                self.sigma[c, :, :] = this_sigma

            # update the inverse of the covariance matrices for future interations
            self.inverse_sigma = np.linalg.inv(self.sigma)

            assert_allclose(np.sum(self.pi), 1)

def log_gaussian_pdf(x_i: np.ndarray, mu: float, inverse_sigma: float) -> float:
    """
    Compute log N(x_i | mu, sigma)
    """
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(inverse_sigma)
    b = 1/b

    # calculate the inverse of sigma
    y = inverse_sigma @ (x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)

def logsumexp(log_probs, axis=None) -> float:
    """
    logF1, logF2, logF3, ... -> log(F1 + F2 + F3 + ...)
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
