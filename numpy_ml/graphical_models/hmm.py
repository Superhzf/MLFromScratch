import numpy as np

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

        Parameters:
        -----------------------
        hidden_states: int
            The number of unique hidden states. It is the only required input
            parameter.
        symbols: int
            The number of unique observation types.
        A: numpy.ndarray of shape (N, N)
            The transmission matrix between hidden states. For example, A[i, j]
            gives the probability from state i to state j.
        B: numpy.ndarray of shape (N, V)
            The emission matrix. For example, B[i, j] gives the probability to
            observe j given the state i.
        pi: numpy.ndarray of shape (N, )
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
        self.eps = np.finfo(float).eps
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
            self.A = []
            for _ in range(self.hidden_states):
                this_A = np.random.dirichlet(np.ones(self.hidden_states),size=1)
                self.A.append(this_A)
            self.A = np.array(self.A)

        # Initialize self.symbols
        if self.symbols is None:
            self.symbols = np.max(self.X)+1

        # Initialize self.B
        if self.B is None:
            self.B = []
            for _ in range(self.hidden_states):
                this_B = np.random.dirichlet(np.ones(self.symbols),size=1)
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
        assert np.allclose(self.A.sum(axis=1), np.ones(1, self.hidden_states)),\
        "The sum of the transmission matrix along any axis should be 1."

        # check self.symbols and self.B
        assert np.allclose(self.B.sum(axis=1), np.ones(1, self.hidden_states)),\
        "The sum of the emission matrix for each state should be 1."
        assert np.B.shape[1] == self.symbols,\
        "The number of columns of the emission matrix should equal to the \
        number of observation types"

        # check self.pi
        assert np.allclose(self.pi.sum(), 1.0),"The prior probability of \
        hiddens should equal 1"

        # check the input X
        assert np.min(self.X) == 0
        assert self.max(self.X) + 1 <= self.symbols

    def fit(self, X: list) -> None:
        """
        Estimate parameters A, B, and pi given observations X. Note that different
        observations in X could have different lengths.

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
        prev_log_ll = 0
        for this_x in X:
            log_ll_prev+=log_likelihood(this_x)

        # for _ in range(self.max_iter):
        #     self.n_iter+=1
        #     this_log_ll =
        #     if abs(this_log_ll - prev_log_ll) <= self.tol:
        #         self.is_converged = True
        #         break

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
        is the probability at any certain time k that the hidden state is Zt given
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


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
