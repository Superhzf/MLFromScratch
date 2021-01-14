import sys
sys.path.append('..')
import numpy as np
from numpy_ml.graphical_models.hmm import DiscreteHMM, GaussHMM
from sklearn.mixture import GaussianMixture
from numpy.testing import assert_almost_equal
from hmmlearn.hmm import MultinomialHMM, GaussianHMM


def test_DiscreteHMM_forward(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    while i < cases:
        tol=1e-3
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        # symbols is the number of unqiue observation types.
        symbols = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            # the actual length is seq_length + 1
            seq_length = symbols
            this_x = np.random.choice(range(symbols), size=seq_length, replace=False)
            X.append(this_x)
            lengths.append(seq_length)
        max_iter = 100


        hmm_gold = MultinomialHMM(n_components=hidden_states, n_iter=100, tol=tol)
        X_gold = np.concatenate(X).reshape((-1,1))
        hmm_gold.fit(X_gold, lengths)
        gold_A = hmm_gold.transmat_
        gold_B = hmm_gold.emissionprob_
        gold_pi = hmm_gold.startprob_
        gold_log_prob = hmm_gold.score(X_gold, lengths)

        hmm_mine = DiscreteHMM(hidden_states=hidden_states,
                               symbols=symbols,
                               A=gold_A,
                               B=gold_B,
                               pi=gold_pi)
        mine_log_prob_list = [hmm_mine.log_likelihood(this_x) for this_x in X]
        mine_log_prob = sum(mine_log_prob_list)
        assert_almost_equal(mine_log_prob, gold_log_prob, decimal=N_decimal)

        i+=1
    print ("Successfully testing the forward algorithm in discrete HMM!")

def test_DiscreteHMM_posteriors(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    while i < cases:
        tol=1e-3
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        # symbols is the number of unqiue observation types.
        symbols = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            # the actual length is seq_length + 1
            seq_length = symbols
            this_x = np.random.choice(range(symbols), size=seq_length, replace=False)
            X.append(this_x)
            lengths.append(seq_length)
        max_iter = 100


        hmm_gold = MultinomialHMM(n_components=hidden_states, n_iter=100, tol=tol)
        X_gold = np.concatenate(X).reshape((-1,1))
        hmm_gold.fit(X_gold, lengths)
        gold_A = hmm_gold.transmat_
        gold_B = hmm_gold.emissionprob_
        gold_pi = hmm_gold.startprob_
        _, gold_posteriors = hmm_gold.score_samples(X_gold, lengths)

        hmm_mine = DiscreteHMM(hidden_states=hidden_states,
                               symbols=symbols,
                               A=gold_A,
                               B=gold_B,
                               pi=gold_pi)
        mine_posteriors = [hmm_mine.posterior(this_x) for this_x in X]
        mine_posteriors = np.concatenate(mine_posteriors)
        assert_almost_equal(mine_posteriors, gold_posteriors, decimal=N_decimal)
        i+=1

    print('Successfully testing the posterior function in discrete HMM!')

def test_DiscreteHMM_decode(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    while i < cases:
        tol=1e-3
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        # symbols is the number of unqiue observation types.
        symbols = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            # the actual length is seq_length + 1
            seq_length = symbols
            this_x = np.random.choice(range(symbols), size=seq_length, replace=False)
            X.append(this_x)
            lengths.append(seq_length)
        max_iter = 100


        hmm_gold = MultinomialHMM(n_components=hidden_states, n_iter=100, tol=tol)
        X_gold = np.concatenate(X).reshape((-1,1))
        hmm_gold.fit(X_gold, lengths)
        gold_A = hmm_gold.transmat_
        gold_B = hmm_gold.emissionprob_
        gold_pi = hmm_gold.startprob_
        gold_logprob, gold_state_sequence = hmm_gold.decode(X_gold, lengths)
        hmm_mine = DiscreteHMM(hidden_states=hidden_states,
                               symbols=symbols,
                               A=gold_A,
                               B=gold_B,
                               pi=gold_pi)
        mine_logprob_list = []
        mine_state_sequence = []
        for this_x in X:
            this_mine_logprob, this_mine_state_sequence = hmm_mine.decode(this_x)
            mine_logprob_list.append(this_mine_logprob)
            mine_state_sequence.append(this_mine_state_sequence)
        mine_state_sequence = np.concatenate(mine_state_sequence)
        mine_logprob = sum(mine_logprob_list)
        assert_almost_equal(mine_logprob, gold_logprob, decimal=N_decimal)
        assert_almost_equal(mine_state_sequence, gold_state_sequence, decimal=N_decimal)
        i+=1
    print('Successfully testing the function of computing decodes in discrete HMM!')


def test_DiscreteHMM_fit(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        # symbols is the number of unqiue observation types.
        symbols = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            # the actual length is seq_length + 1
            seq_length = symbols
            this_x = np.random.choice(range(symbols), size=seq_length, replace=False)
            X.append(this_x)
            lengths.append(seq_length)

        A = np.full((hidden_states, hidden_states),1/hidden_states)

        B = []
        for _ in range(hidden_states):
            this_B = np.random.dirichlet(np.ones(symbols),size=1)[0]
            B.append(this_B)
        B = np.array(B)

        pi = np.ones(hidden_states)
        pi = pi/hidden_states


        hmm_gold = MultinomialHMM(n_components=hidden_states,
                                  startprob_prior=1,
                                  transmat_prior=1,
                                  init_params='',
                                  n_iter=max_iter,
                                  tol=tol)
        hmm_gold.transmat_ = A
        hmm_gold.emissionprob_ = B
        hmm_gold.startprob_ = pi

        X_gold = np.concatenate(X).reshape((-1,1))
        hmm_gold.fit(X_gold, lengths)

        gold_A = hmm_gold.transmat_
        gold_B = hmm_gold.emissionprob_
        gold_pi = hmm_gold.startprob_

        hmm_mine = DiscreteHMM(hidden_states=hidden_states,
                               symbols=symbols,
                               A=A,
                               B=B,
                               pi=pi,
                               tol=tol,
                               max_iter=max_iter)
        hmm_mine.fit(X)
        mine_A = hmm_mine.A
        mine_B = hmm_mine.B
        mine_pi = hmm_mine.pi
        assert_almost_equal(mine_pi, gold_pi, decimal=N_decimal)
        assert_almost_equal(mine_A, gold_A, decimal=N_decimal)
        assert_almost_equal(mine_B, gold_B, decimal=N_decimal)
        i+=1

    print('Successfully testing the function of estimating parameters in discrete HMM!')


def test_GaussHMM_forward(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        n_features = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            seq_length = np.random.randint(4, 9)
            this_x = np.random.rand(seq_length,n_features)

            X.append(this_x)
            lengths.append(seq_length)

        hmm_gold = GaussianHMM(n_components=hidden_states,
                               covariance_type='full',
                               n_iter=max_iter,
                               tol=tol)

        X_gold = np.concatenate(X)
        hmm_gold.fit(X_gold, lengths)

        gold_means = hmm_gold.means_
        gold_pi = hmm_gold.startprob_
        gold_n_features = hmm_gold.n_features
        gold_transmat = hmm_gold.transmat_
        gold_means = hmm_gold.means_
        gold_covars = hmm_gold.covars_

        hmm_mine = GaussHMM(hidden_states=hidden_states,
                               A=gold_transmat,
                               n_features=gold_n_features,
                               means=gold_means,
                               covar=gold_covars,
                               pi=gold_pi,
                               tol=tol,
                               max_iter=max_iter)
        gold_log_likelihood = hmm_gold.score(X_gold, lengths)

        mine_ll_list = [hmm_mine.log_likelihood(this_x) for this_x in X]
        mine_log_likelihood = sum(mine_ll_list)

        assert_almost_equal(mine_log_likelihood, gold_log_likelihood, decimal=N_decimal)
        i+=1

    print('Successfully testing the forward algorithm in Gaussian HMM!')


def test_GaussHMM_posterior(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        n_features = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            seq_length = np.random.randint(4, 9)
            this_x = np.random.rand(seq_length,n_features)

            X.append(this_x)
            lengths.append(seq_length)

        hmm_gold = GaussianHMM(n_components=hidden_states,
                               covariance_type='full',
                               n_iter=max_iter,
                               tol=tol)

        X_gold = np.concatenate(X)
        hmm_gold.fit(X_gold, lengths)

        gold_means = hmm_gold.means_
        gold_pi = hmm_gold.startprob_
        gold_n_features = hmm_gold.n_features
        gold_transmat = hmm_gold.transmat_
        gold_means = hmm_gold.means_
        gold_covars = hmm_gold.covars_

        hmm_mine = GaussHMM(hidden_states=hidden_states,
                               A=gold_transmat,
                               n_features=gold_n_features,
                               means=gold_means,
                               covar=gold_covars,
                               pi=gold_pi,
                               tol=tol,
                               max_iter=max_iter)
        _,gold_posteriors = hmm_gold.score_samples(X_gold, lengths)
        mine_posterior_list = [hmm_mine.posterior(this_x) for this_x in X]
        mine_posterior_list = np.concatenate(mine_posterior_list)
        assert_almost_equal(mine_posterior_list, gold_posteriors, decimal=N_decimal)
        i+=1

    print('Successfully testing the function of computing posteriors in Gaussian HMM!')


def test_GaussHMM_decode(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        hidden_states = np.random.randint(3, 6)
        n_features = np.random.randint(4, 9)
        X = []
        lengths = []
        for _ in range(n_samples):
            seq_length = np.random.randint(4, 9)
            this_x = np.random.rand(seq_length,n_features)

            X.append(this_x)
            lengths.append(seq_length)

        hmm_gold = GaussianHMM(n_components=hidden_states,
                               covariance_type='full',
                               algorithm='viterbi',
                               n_iter=max_iter,
                               tol=tol)

        X_gold = np.concatenate(X)
        hmm_gold.fit(X_gold, lengths)

        gold_means = hmm_gold.means_
        gold_pi = hmm_gold.startprob_
        gold_n_features = hmm_gold.n_features
        gold_transmat = hmm_gold.transmat_
        gold_means = hmm_gold.means_
        gold_covars = hmm_gold.covars_

        hmm_mine = GaussHMM(hidden_states=hidden_states,
                               A=gold_transmat,
                               n_features=gold_n_features,
                               means=gold_means,
                               covar=gold_covars,
                               pi=gold_pi,
                               tol=tol,
                               max_iter=max_iter)
        gold_logprob,gold_state_seq = hmm_gold.decode(X_gold, lengths)
        mine_logprob_list = []
        mine_state_seq_list = []
        for this_x in X:
            this_logprob, this_state_seq = hmm_mine.decode(this_x)
            mine_logprob_list.append(this_logprob)
            mine_state_seq_list.append(this_state_seq)
        mine_logprob = sum(mine_logprob_list)
        mine_state_seq = np.concatenate(mine_state_seq_list)
        assert_almost_equal(mine_logprob, gold_logprob, decimal=N_decimal)
        assert_almost_equal(mine_state_seq, gold_state_seq, decimal=N_decimal)
        i+=1

    print('Successfully testing the decode function in Gaussian HMM!')
