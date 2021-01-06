import sys
sys.path.append('..')
import numpy as np
from numpy_ml.graphical_models.hmm import DiscreteHMM
from sklearn.mixture import GaussianMixture
from numpy.testing import assert_almost_equal
from hmmlearn.hmm import MultinomialHMM


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
    print ("Successfully testing the forward algorithm in Discrete HMM!")

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

    print('Successfully testing the posterior function in HMM!')

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
    print('Successfully testing the function of computing decodes!')

        
