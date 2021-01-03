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
