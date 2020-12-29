import sys
sys.path.append('..')
import numpy as np
from numpy_ml.unsupervised_learning.GMM import GMM
from sklearn.mixture import GaussianMixture
from numpy.testing import assert_almost_equal
from .helpers import random_tensor

def test_GMM(cases: str) -> None:
    """
    Please note that the test is conducted if and only if there are no singular
    matrices problems.
    """
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 2
    while i < cases:
        n_samples = np.random.randint(2, 100)
        n_clsuters = np.random.randint(2, 10)
        n_dim = np.random.randint(2, 10)
        X = random_tensor((n_samples, n_dim), standardize=False)
        max_iter = 100
        tol = 1e-3

        # initialize weights
        weights_init = np.random.rand(n_clsuters)
        weights_init = weights_init/weights_init.sum()

        # initialize means
        means_init = np.zeros([n_clsuters, n_dim])
        for this_dim in range(n_dim):
            this_mu = np.random.choice(X[:,this_dim], n_clsuters)
            means_init[:, this_dim] = this_mu

        # initialize inverse sigma
        precisions_init = np.array([np.identity(n_dim) for _ in range(n_clsuters)])


        gmm_gold = GaussianMixture(n_components=n_clsuters,
                             covariance_type = 'full',
                             tol=tol,
                             reg_covar=0,
                             max_iter=max_iter,
                             weights_init=weights_init,
                             means_init=means_init,
                             precisions_init=precisions_init)
        gmm_mine = GMM(C=n_clsuters,
                       seed=None,
                       max_iter=max_iter,
                       tol=tol,
                       weights_init=weights_init,
                       means_init=means_init,
                       precisions_init=precisions_init)

        try:
            gmm_gold.fit(X)
        except Exception as e:
            continue

        gmm_mine.fit(X)

        gold_weights_ = gmm_gold.weights_
        gold_means_ = gmm_gold.means_
        gold_sigma_ = gmm_gold.covariances_
        gold_n_iter_ = gmm_gold.n_iter_
        gold_lower_bound_ = gmm_gold.lower_bound_

        mine_weights_ = gmm_mine.best_pi
        mine_means_ = gmm_mine.best_mu
        mine_sigma_ = gmm_mine.best_sigma
        mine_lower_bound = gmm_mine.best_elbo
        mine_iter = gmm_mine.n_iter_

        assert_almost_equal(mine_weights_, gold_weights_, decimal=N_decimal)
        assert_almost_equal(mine_means_, gold_means_, decimal=N_decimal)
        assert_almost_equal(mine_sigma_, gold_sigma_, decimal=N_decimal)
        assert_almost_equal(mine_lower_bound, gold_lower_bound_, decimal=N_decimal)
        assert_almost_equal(mine_iter, gold_n_iter_, decimal=N_decimal)
        i+=1
