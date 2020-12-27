import numpy as np

class GMM(object):
    def __init__(self, C:int=3, seed:int=None) -> None:
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

        Attributes:
        ----------------------
        N: int
            The number of observations in the training set.
        d: int
            The dimension of each observation in the training set.
        pi: numpy.array of shape (C,)
            The cluster priors. pi[j] shows the probability that the latent
            variable comes from the Gaussian distribution j.
        Q: numpy.array of shape of shape (N, C)
            The distribution over the laten variable Z.
        mu: numpy.array of shape (C, d)
            Means of Gaussian distributions
        sigma: numpy.array of shape (C, d, d)
            Covariance matrices for Gaussian distributions
        """
        self.C = C
        self.N = N
        self.d = None
        self.seed = seed

        if self.seed:
            np.random.seed(self.seed)

        def _initialize_parameters(self):
            self.pi = np.random.rand(self.C)
            self.pi = self.pi/self.pi.sum()

            self.Q = np.zeros([self.N, self.C])
            # Initialize mu by randomly select C values for each dimension
            self.mu = np.zeros([self.C, self.d])
            for this_dim in range(self.d):
                this_mu = np.random.choice(self.X[:,this_dim],self.C)
                self.mu[:, this_dim] = this_mu
            # Diag covariance matrices
            self.sigma = np.array([np.identity(d) for _ in range(C)])

            self.best_pi = None
            self.best_mu = None
            self.best_sigma = None
            self.best_elbo = -np.inf

        def likelihood_lower_bound(self):
            """Calculate the ELBO under the current GMM parameters"""
            eps = np.finfo(float).eps
            elbo = 0

            for i in range(sef.N):
                x_i = self.X[i]

                for this_c in range(C):
                    pi_c = self.pi[this_c]
                    z_c = self.Q[i, this_c]
                    mu_c = self.mu[this_c, :]
                    sigma_c = self.sigma[this_c, :, :]

                    log_pi_c = np.log(pi_c+eps)
                    log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)
                    log_z_c = np.log(z_c+eps)
                    elbo += z_c * (log_pi_c + log_p_x_i - log_z_c)
            return elbo

        def fit(self, X: , max_iter:int=100, tol:float=1e-3):
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

            """
            self.X = X
            self.N = X.shape[0]
            self.d = X.shape[1]

            self._initialize_parameters()
            prev_elbo = -np.inf

            for _ in range(max_iter):
                self._E_step()
                self._M_step()

        def _E_step(self):
            for i in range(self.N):
                x_i = self.X[i, :]

                denom_vals = []
