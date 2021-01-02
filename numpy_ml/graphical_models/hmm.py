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
            The number
        """
        self.hidden_states = hidden_states
        self.symbols = symbols
        self.eps = np.finfo(float).eps
        self.A = A
        self.B = B
        self.pi = pi
        self.seed = seed
        self.max_iter = max_iter
        self.n_iter = 0

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
                self.B.append(self.this_B)
            self.B = np.array(self.B)

        # Initialize self.pi
        if self.pi is None:
            self.pi = np.ones(self.hidden_states)
            self.pi = self.pi/self.hidden_states

    def _parameter_check(self) -> None:
        # check self.hidden_states and self.A
        assert self.hidden_states == self.A.shape[0],
        "The input number of hidden states does not equal to the shape of A."
        assert self.A.shape[0] == self.A.shape[1],
        "The number of columns and rows for A should be the same"
        assert np.allclose(self.A.sum(axis=1), np.ones(1, self.hidden_states)),
        "The sum of the transmission matrix along any axis should be 1."

        # check self.symbols and self.B
        assert np.allclose(self.B.sum(axis=1), np.ones(1, self.hidden_states)),
        "The sum of the emission matrix for each state should be 1."
        assert np.B.shape[1] == self.symbols,
        "The number of columns of the emission matrix should equal to the \
        number of observation types"

        # check self.pi
        assert np.allclose(self.pi.sum(), 1.0),"The prior probability of \
        hiddens should equal 1"

        # check the input X
        assert np.min(self.X) == 0
        assert self.max(self.X) + 1 <= self.symbols

    def fit(self, X: list) -> None:
        self.X = X
        self.I = len(self.X)

        self._initialize()
        self._parameter_check()

    def likelihood(self, x: np.ndarray) -> float:
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
        forward = self._forward(x)

    def _forward(self, x) -> np.ndarray:
        """
        Parameters:
        ----------------
        x: numpy.ndarray of shape (T, ).
            A single set of observations. Note that T is not the same for
            different observations.
        """
        T = x.shape[0]
        alpha_it = np.zeros((self.hidden_states, T))
        # initialization
        alpha_it[:,0] = self.pi * self.B[:,x[0]]

        for this_t in range(1, T):
            this_obs = x[this_t]
            for this_state_prev in range(self.hidden_states):
                for this_state_next in range(self.hidden_states):
                    alpha_it[this_state, this_t] = self.A[this_t-1, this_t] * \
                                                   self.B[this_t, this_obs] * \
                                                   alpha_it[this_state]
