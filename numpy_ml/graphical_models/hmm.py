import numpy as np


"""
WARNING:

The formula for updating Aij and Bij in the cs229 notes is not correct.
"""
class DiscreteHMM:
    def __init__(self,
                 hidden_states:int=1,
                 symbols:int=None,
                 A:np.ndarray=None,
                 B:np.ndarray=None,
                 pi:np.ndarray=None,
                 seed:int=None,
                 tol:float=1e-3,
                 max_iter:int=100)->None:
        """
        A Hidden Markov Model with multinomial discrete emission distributions.

        Ref: http://cs229.stanford.edu/section/cs229-hmm.pdf
        https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

        Parameters:
        -----------------------
        hidden_states: int
            The number of unique hidden states. It is the only required input
            parameter for a HMM model.
        symbols: int
            The number of unique observation types.
        A: numpy.ndarray of shape (hidden_states, hidden_states)
            The transmission matrix between hidden states. For example, A[i, j]
            gives the probability from state i to state j.
        B: numpy.ndarray of shape (hidden_states, symbols)
            The emission matrix. For example, B[i, j] gives the probability to
            observe j given the state i.
        pi: numpy.ndarray of shape (hidden_states, )
            The prior probability of hidden states.
        seed: int
            The seed used to randomly generate A and B is not provided.
        tol: float
            The tolerance value. If the difference in log likelihood between
            two epochs is less than this value, terminate training.
        max_iter: int
            The maximum number of iterations to estimate A and B.

        Attributes:
        -----------------------
        X: list-like of shape (I, )
            The set of observed training sequences. Note that the length of
            different observations could be different. For each observation, the
            variable is supposed to be label encoded. For example, suppose one
            observation has a length of 3, and 4 possible symbols, typically it
            may look like:
            [[1,0,0,0],
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]]
            However, here this observation should be encoded as [0,0,1,2].
        I: int
            The number of observations.
        n_iter:
            The number of rounds used to estimate A, B and pi.
        is_converged: bool
            It indicates whether the model is converged when estimating A, B and
            pi.
        """
        self.hidden_states = hidden_states
        self.symbols = symbols
        self.A = A
        self.B = B
        self.pi = pi
        self.seed = seed
        self.tol = tol
        self.max_iter = max_iter
        self.n_iter = 0
        self.is_converged=False

    def _initialize(self) -> None:
        "Initialize parameters and do parameter check"
        if self.seed:
            np.random.seed(self.seed)
        # Initialize self.A
        if self.A is None:
            self.A = np.full((self.hidden_states, self.hidden_states),1/self.hidden_states)

        # Initialize self.symbols
        if self.symbols is None:
            self.symbols = np.max(self.X)+1

        # Initialize self.B
        if self.B is None:
            self.B = []
            for _ in range(self.hidden_states):
                this_B = np.random.dirichlet(np.ones(self.symbols),size=1)[0]
                self.B.append(this_B)
            self.B = np.array(self.B)

        # Initialize self.pi
        if self.pi is None:
            self.pi = np.ones(self.hidden_states)
            self.pi = self.pi/self.hidden_states

    def _parameter_check(self) -> None:
        # check self.hidden_states and self.A
        assert self.hidden_states == self.A.shape[0],\
        "The input number of hidden states does not equal to the shape of A."
        assert self.A.shape[0] == self.A.shape[1],\
        "The number of columns and rows for A should be the same"
        assert np.allclose(self.A.sum(axis=1), np.ones((1, self.hidden_states))),\
        "The sum of the transmission matrix along any axis should be 1."

        # check self.symbols and self.B
        assert np.allclose(self.B.sum(axis=1), np.ones((1, self.hidden_states))),\
        "The sum of the emission matrix for each state should be 1."
        assert self.B.shape[1] == self.symbols,\
        "The number of columns of the emission matrix should equal to the \
        number of observation types"

        # check self.pi
        assert np.allclose(self.pi.sum(), 1.0),"The prior probability of \
        hiddens should equal 1"

        # check the input X
        assert np.min(self.X) == 0
        assert np.max(self.X) + 1 <= self.symbols

    def fit(self, X: list) -> None:
        """
        Estimate parameters A, B, and pi given observations X via Baum Welch
        algorithm. Note that different observations in X could have different
        lengths.

        Parameters:
        ---------------------
        X: list of length I.
            Observations used to estimate A, B and pi. Please refer to the comments
            under the class definition for more details.
        """
        self.X = X
        self.I = len(self.X)

        self._initialize()
        self._parameter_check()
        log_ll_monitor = np.zeros(2)

        for _ in range(self.max_iter):
            self.n_iter+=1
            this_log_ll = 0
            gamma, xi, phi = self._Estep()
            #There is no specific reason why this_log_ll is calculated between
            # Estep and Mstep, I do this in order to pass the unit test with
            # hmmlearn
            for this_x in X:
                this_log_ll+=self.log_likelihood(this_x)

            self.A, self.B, self.pi = self._Mstep(gamma, xi, phi)
            log_ll_monitor[0] = log_ll_monitor[1]
            log_ll_monitor[1] = this_log_ll
            if log_ll_monitor[1]*log_ll_monitor[0] !=0 and \
                        log_ll_monitor[1]-log_ll_monitor[0]<self.tol:
                self.is_converged=True
                break
        return

    def _Estep(self) -> None:
        """
        Run a singl E step for the Baum-Welch algorithm.
        """
        gamma = []
        xi = []
        pi = np.zeros((self.I, self.hidden_states))
        for idx, x in enumerate(self.X):
            alpha_it = self._forward(x)
            beta = self._backward(x)
            T = len(x)
            this_gamma = np.zeros((self.hidden_states, T))
            this_xi = np.zeros((self.hidden_states, self.hidden_states, T-1))
            xi_buffer = np.zeros(self.hidden_states)
            this_phi = np.zeros(self.hidden_states)

            for this_s_prev in range(self.hidden_states):
                this_gamma[this_s_prev, T-1] = alpha_it[this_s_prev,T-1]+\
                                                    beta[T-1, this_s_prev]
            this_gamma[:, T-1] = this_gamma[:, T-1] - logsumexp(this_gamma[:, T-1])

            for this_t in range(T-1):
                obs = x[this_t+1]
                for s_prev in range(self.hidden_states):
                    this_gamma[s_prev, this_t] = alpha_it[s_prev,this_t]+\
                                                        beta[this_t, s_prev]
                    for s_next in range(self.hidden_states):
                        this_xi[s_prev, s_next, this_t]=alpha_it[s_prev, this_t]+\
                                                        np.log(self.A[s_prev, s_next])+\
                                                        beta[this_t+1, s_next]+\
                                                        np.log(self.B[s_next, obs])
                    xi_buffer[s_prev]=logsumexp(this_xi[s_prev,:, this_t])

                this_gamma[:, this_t] = this_gamma[:, this_t] - logsumexp(this_gamma[:, this_t])

                this_xi[:, :, this_t] = this_xi[:, :, this_t] - logsumexp(xi_buffer)

            gamma.append(this_gamma)
            xi.append(this_xi)

            pi[idx]=this_gamma[:,0]
        return gamma, xi, pi

    def _Mstep(self, gamma: list, xi: list, pi: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        new_A = np.zeros((self.hidden_states, self.hidden_states))
        new_B = np.zeros((self.hidden_states, self.symbols))
        new_pi = np.zeros(self.hidden_states)

        count_gamma = np.zeros((self.I, self.hidden_states, self.symbols))
        count_xi = np.zeros((self.I, self.hidden_states, self.hidden_states))

        for idx, x in enumerate(self.X):
            for s_prev in range(self.hidden_states):
                for vk in range(self.symbols):
                    if (x != vk).all():
                        count_gamma[idx, s_prev, vk] = -np.inf
                    else:
                        count_gamma[idx, s_prev, vk] = logsumexp(gamma[idx][s_prev, x==vk])
                for s_next in range(self.hidden_states):
                    count_xi[idx, s_prev, s_next] = logsumexp(xi[idx][s_prev, s_next, :])
        new_pi = logsumexp(pi,axis=0)-np.log(self.I)
        np.testing.assert_almost_equal(np.exp(new_pi).sum(), 1)

        for s_prev in range(self.hidden_states):
            for vk in range(self.symbols):
                new_B[s_prev, vk] = logsumexp(count_gamma[:, s_prev, vk]) - \
                                            logsumexp(count_gamma[:, s_prev, :])

            for s_next in range(self.hidden_states):
                new_A[s_prev, s_next] = logsumexp(count_xi[:, s_prev, s_next]) -\
                                        logsumexp(count_xi[:, s_prev, :])

            np.testing.assert_almost_equal(np.exp(new_A[s_prev, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(new_B[s_prev, :]).sum(), 1)
        return np.exp(new_A), np.exp(new_B), np.exp(new_pi)

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        Given A, B, pi, and a set of observations, compute the probability of
        observations.

        The likelhood is calculated via the forward algorithm.

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ---------------
        likelihood: float
            The likelihood of the observation.
        """
        alpha_it = self._forward(x)
        log_likelihood = logsumexp(alpha_it[:,-1])
        return log_likelihood

    def posterior(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the posteriors. P(Zt|X) (not the log one.).

        Forward: P(Zt, X[1:t])
        Backward: P(X[t+1:T]|Zt)

        P(Zt|X) = P(Zt, X)/P(X) = P(X[t+1:T]|Zt)*P(Zt, X[1:t])/P(X)

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. T is the sequence length. Note that T
            is not the same for different observations.

        Return:
        posteriors: numpy.ndarray of shape (T, hidden_states)
            posteriors[t, i] gives P(Zt=si|X)
        """
        T = len(x)
        posteriors = np.zeros((T, self.hidden_states))
        forward = self._forward(x)
        backward = self._backward(x)
        for this_t in range(T):
            for this_s in range(self.hidden_states):
                this_posterior = forward[this_s, this_t] + backward[this_t, this_s]
                normalizer = self.log_likelihood(x)
                this_posterior = this_posterior - normalizer
                posteriors[this_t, this_s] = np.exp(this_posterior)
        return posteriors

    def decode(self, x: np.ndarray) -> [float, np.ndarray]:
        """
        Given A, B, pi and the input x, compute the most probable sequence of
        latent states via Viterbi algorithm and its log probability.

        viterbi[i,j] gives the log probability of Zj given X1:Xi.
        path_track[i,j] gives which which state Zj returns the viterbi[i,j].

        Parameters:
        -------------------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ------------------------
        best_path_log_prob: float
            The probability of the latent state sequence in best_path
        best_path: numpy.ndarray of shape (T,)
            The most probable sequence of laten states for the observation.
        """
        T = len(x)
        viterbi = np.zeros((T, self.hidden_states))
        path_track = np.zeros((T, self.hidden_states))

        viterbi[0, :] = np.log(self.pi) + np.log(self.B[:, x[0]])

        work_buffer =np.zeros(self.hidden_states)
        for this_t in range(1, T):
            this_x = x[this_t]
            for this_s_next in range(self.hidden_states):
                for this_s_prev in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_s_prev] = np.log(self.A[this_s_prev, this_s_next])+\
                                                   viterbi[this_t-1, this_s_prev]
                with np.errstate(divide="ignore"):
                    viterbi[this_t, this_s_next]=np.max(work_buffer) + np.log(self.B[this_s_next, this_x])
                path_track[this_t, this_s_next] = np.argmax(work_buffer)

        best_path_log_prob = np.max(viterbi[T-1,:])

        best_path=[]
        pointer = viterbi[T-1,:].argmax()
        best_path.append(pointer)
        for this_t in reversed(range(1, T)):
            pointer = path_track[this_t, pointer]
            pointer = int(pointer)
            best_path.append(pointer)
        best_path=np.array(best_path[::-1])
        return best_path_log_prob, best_path


    def _forward(self, x:np.ndarray) -> np.ndarray:
        """
        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. Note that T is not the same for
            different observations.

        Return:
        ---------------
        alpha_it: numpy.ndarray of shape (hidden_states, T)
            alpha_it[i,t] gives logP(Zt=Si, X[1:t])
        """
        T = x.shape[0]
        alpha_it = np.zeros((self.hidden_states, T))
        # initialization
        with np.errstate(divide="ignore"):
            alpha_it[:,0] = np.log(self.pi) + np.log(self.B[:,x[0]])

        work_buffer = np.zeros(self.hidden_states)
        for this_t in range(1, T):
            this_obs = x[this_t]
            for this_state in range(self.hidden_states):
                for this_state_prev in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_state_prev]=np.log(self.A[this_state_prev, this_state]) + \
                                                     alpha_it[this_state_prev, this_t-1]
                with np.errstate(divide="ignore"):
                    alpha_it[this_state, this_t]=logsumexp(work_buffer) + \
                                                np.log(self.B[this_state, this_obs])

        return alpha_it

    def _backward(self, x) -> np.ndarray:
        """
        Given A, B and pi, compute beta[i, t] = logP(X_t+1,...,X_T|Z_t=si).

        The motivation to compute this probability is to asnwer the question: what
        is the probability at any certain time t that the hidden state is Zt given
        the sequence of observations, which is P(Zt|X).

        Forward: P(Zt, X[1:t])
        Backward: P(X[t+1:T]|Zt)

        P(Zt|X) = P(Zt, X)/P(X) = P(X[t+1:T]|Zt)*P(Zt, X[1:t])/P(X)

        Instead of the direct probability, log-probability is computed here.

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ----------------
        beta: numpy.arary of shape (T, hidden_states)
            beta[t, i] gives logP(X[t+1:T]|Zt=Si)
        """
        T = len(x)
        beta = np.zeros((T, self.hidden_states))
        # Explicitly set up beta[T-1, :] = log1 = 0
        beta[-1, :] = np.zeros(self.hidden_states)

        work_buffer = np.zeros(self.hidden_states)
        for this_t in reversed(range(T-1)):
            next_obs = x[this_t+1]
            for this_s_prev in range(self.hidden_states):
                for this_s_next in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_s_next] = beta[this_t+1, this_s_next]+\
                                                   np.log(self.B[this_s_next, next_obs])+\
                                                   np.log(self.A[this_s_prev, this_s_next])
                with np.errstate(divide="ignore"):
                    beta[this_t, this_s_prev] = logsumexp(work_buffer)
        return beta


class GaussHMM:
    def __init__(self,
                 hidden_states:int=1,
                 A:np.ndarray=None,
                 n_features:int=None,
                 means: np.ndarray=None,
                 covar:np.ndarray=None,
                 pi:np.ndarray=None,
                 seed:int=None,
                 tol:float=1e-3,
                 max_iter:int=100)->None:
        """
        A Hidden Markov Model with Gaussian emission distributions. Specifically,
        the input X is allowed to be continuous, although the hidden states are
        still supposed to be discrete.

        Parameters:
        -------------------------------
        hidden_states: int
            The number of unique hidden states
        A: numpy.ndarray of shape (hidden_states, hidden_states)
            The transmission matrix between states. A[i,j] gives the probability
            from state i to state j.
        n_features: int
            The number of features each observation has.
        means: numpy.ndarray of shape (hidden_states,n_features)
            It is the mean parameter for the distribution that returns the
            probability of x given the mean and covariance.
        covar: numpy.ndarray of shape (hidden_states, n_features, n_features)
            It is the covariance parameter for the distribution that returns
            the probability of x given the mean and covariance
        pi: numpy.ndarray of shape (hidden_states,)
            It is the prior probability of hidden states
        seed: int
            It is the seed to initialize parameters if not provided
        tol: float
            The minimum required improvement between two iterations when
            estimating parameters
        max_iter: int
            The maximum number of iterations when estimating parameters.

        Attributes:
        -------------------
        X: list-like of shape (I,)
            The training set with N observations. Earch observation is a numpy
            ndarray of shape (I, n_features). Each observation may have a
            differett I.
        I: int
            The number of observations in the training set
        n_iter:
            The number of rounds used to estimate parameters
        is_converged: bool
            Whether estimating parameters is converged.
        """
        self.hidden_states=hidden_states
        self.A=A
        self.means=means
        self.covar=covar
        self.n_features=n_features
        self.pi=pi
        self.seed=seed
        self.tol=tol
        self.max_iter=max_iter

    def _initialize(self) -> None:
        "Initialize parameters and do parameter check"
        if self.seed:
            np.random.seed(self.seed)
        # Initialize A
        if self.A is None:
            self.A = np.full((self.hidden_states, self.hidden_states),1/self.hidden_states)

        # initialize n_features
        if self.n_features is None:
            self.n_features = self.X[0].shape[1]

        if self.means is None:
            # hmmlearn uses kmeans clustering to initialize
            # https://github.com/hmmlearn/hmmlearn/blob/master/lib/hmmlearn/hmm.py#L207
            # for simplifity, I will randomly select hidden_states observations
            self.means = np.zeros((self.hidden_states,self.n_features))
            random_obs = np.random.choice(self.X, size=self.hidden_states, replace=False)
            for idx, this_obs in enumerate(random_obs):
                self.means[idx]=this_obs.mean(axis=0)

        # Initialize self.B
        if self.covar is None:
            self.covar = np.zeros((self.hidden_states, self.n_features, self.n_features))
            random_obs = np.random.choice(self.X, size=self.hidden_states, replace=False)
            for idx, this_obs in enumerate(random_obs):
                self.covar[idx]=np.cov(this_obs.T)

        # Initialize self.pi
        if self.pi is None:
            self.pi = np.ones(self.hidden_states)
            self.pi = self.pi/self.hidden_states

    def fit(self,X):
        """
        TODO: I don't think it is meaningful to do this right now, because it
        is just a matter of time. I will skip this part.
        """
        pass

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        Given A, means, covar, pi, and a set of observations, compute the probability of
        observations.

        The likelhood is calculated via the forward algorithm.

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, n_features).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ---------------
        likelihood: float
            The likelihood of the observation.
        """
        alpha_it = self._forward(x)
        log_likelihood = logsumexp(alpha_it[:,-1])
        return log_likelihood

    def posterior(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the posteriors. P(Zt|X) (not the log one.).

        Forward: P(Zt, X[1:t])
        Backward: P(X[t+1:T]|Zt)

        P(Zt|X) = P(Zt, X)/P(X) = P(X[t+1:T]|Zt)*P(Zt, X[1:t])/P(X)

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, n_features).
            A single set of observations. T is the sequence length. Note that T
            is not the same for different observations.

        Return:
        posteriors: numpy.ndarray of shape (T, hidden_states)
            posteriors[t, i] gives P(Zt=si|X)
        """
        T = len(x)
        posteriors = np.zeros((T, self.hidden_states))
        forward = self._forward(x)
        backward = self._backward(x)
        for this_t in range(T):
            for this_s in range(self.hidden_states):
                this_posterior = forward[this_s, this_t] + backward[this_t, this_s]
                normalizer = self.log_likelihood(x)
                this_posterior = this_posterior - normalizer
                posteriors[this_t, this_s] = np.exp(this_posterior)
        return posteriors

    def decode(self, x: np.ndarray) -> [float, np.ndarray]:
        """
        Given A, means and covar, pi and the input x, compute the most probable sequence of
        latent states via Viterbi algorithm and its log probability.

        viterbi[i,j] gives the log probability of Zj given X1:Xi.
        path_track[i,j] gives which state Zj returns the viterbi[i,j].

        Parameters:
        -------------------------
        x: numpy.ndarray of shape (T, n_features).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ------------------------
        best_path_log_prob: float
            The probability of the latent state sequence in best_path
        best_path: numpy.ndarray of shape (T,)
            The most probable sequence of laten states for the observation.
        """
        T = len(x)
        viterbi = np.zeros((T, self.hidden_states))
        path_track = np.zeros((T, self.hidden_states))

        for this_s in range(self.hidden_states):
            viterbi[0, this_s] = np.log(self.pi[this_s]) + log_gaussian_pdf(x[0],
                                                                self.means[this_s],
                                                                self.covar[this_s])

        work_buffer =np.zeros(self.hidden_states)
        for this_t in range(1, T):
            this_x = x[this_t]
            for this_s_next in range(self.hidden_states):
                for this_s_prev in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_s_prev] = np.log(self.A[this_s_prev, this_s_next])+\
                                                   viterbi[this_t-1, this_s_prev]
                with np.errstate(divide="ignore"):
                        viterbi[this_t, this_s_next]=np.max(work_buffer) + log_gaussian_pdf(this_x,
                                                                        self.means[this_s_next],
                                                                        self.covar[this_s_next])
                path_track[this_t, this_s_next] = np.argmax(work_buffer)

        best_path_log_prob = np.max(viterbi[T-1,:])

        best_path=[]
        pointer = viterbi[T-1,:].argmax()
        best_path.append(pointer)
        for this_t in reversed(range(1, T)):
            pointer = path_track[this_t, pointer]
            pointer = int(pointer)
            best_path.append(pointer)
        best_path=np.array(best_path[::-1])
        return best_path_log_prob, best_path

    def _forward(self, x:np.ndarray) -> np.ndarray:
        """
        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, n_features).
            A single set of observations. Note that T is not the same for
            different observations.

        Return:
        ---------------
        alpha_it: numpy.ndarray of shape (hidden_states, T)
            alpha_it[i,t] gives logP(Zt=Si, X[1:t])
        """
        T = x.shape[0]
        alpha_it = np.zeros((self.hidden_states, T))
        # initialization
        with np.errstate(divide="ignore"):
            for this_s in range(self.hidden_states):
                alpha_it[this_s,0] = np.log(self.pi[this_s]) + log_gaussian_pdf(x[0],
                                                                   self.means[this_s,:],
                                                                   self.covar[this_s,:,:])

        work_buffer = np.zeros(self.hidden_states)
        for this_t in range(1, T):
            this_obs = x[this_t]
            for this_state in range(self.hidden_states):
                for this_state_prev in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_state_prev]=np.log(self.A[this_state_prev, this_state]) + \
                                                     alpha_it[this_state_prev, this_t-1]
                with np.errstate(divide="ignore"):
                    alpha_it[this_state, this_t]=logsumexp(work_buffer) + \
                                                log_gaussian_pdf(this_obs,
                                                        self.means[this_state],
                                                        self.covar[this_state,:,:])

        return alpha_it

    def _backward(self, x:np.array)->np.array:
        """
        Given A, means, covar and pi, compute beta[i, t] = logP(X_t+1,...,X_T|Z_t=si).

        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, n_features).
            A single set of observations. Note that T is not the same for
            different observations.

        Returns:
        ----------------
        beta: numpy.arary of shape (T, hidden_states)
            beta[t, i] gives logP(X[t+1:T]|Zt=Si)
        """
        T = len(x)
        beta = np.zeros((T, self.hidden_states))
        # Explicitly set up beta[T-1, :] = log1 = 0
        beta[-1, :] = np.zeros(self.hidden_states)

        work_buffer = np.zeros(self.hidden_states)
        for this_t in reversed(range(T-1)):
            next_obs = x[this_t+1]
            for this_s_prev in range(self.hidden_states):
                for this_s_next in range(self.hidden_states):
                    with np.errstate(divide="ignore"):
                        work_buffer[this_s_next] = beta[this_t+1, this_s_next]+\
                                                   log_gaussian_pdf(next_obs,
                                                           self.means[this_s_next],
                                                           self.covar[this_s_next,:,:])+\
                                                   np.log(self.A[this_s_prev, this_s_next])
                with np.errstate(divide="ignore"):
                    beta[this_t, this_s_prev] = logsumexp(work_buffer)
        return beta


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)

def log_gaussian_pdf(x_i: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute log P(x_i | z_j; mu, sigma)
    """
    d = len(mu)
    a = d * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)
