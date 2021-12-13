import sys
sys.path.append('..')
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import SGDRegressor
from numpy_ml.supervised_learning.regression import RidgeRegression,LassoRegression, LassoRegressionCD
from sklearn.linear_model import Lasso

def test_ridge_regression_SGD(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        n_features = np.random.randint(5,10)
        X = np.random.rand(n_samples,n_features)

        y = np.random.rand(n_samples,)
        alpha = np.random.uniform(low=0.01)
        learning_rate = np.random.uniform(low=0.01,high=0.2)

        # make sure that the initial parameters are the same
        coef_limit=1/np.sqrt(n_features)
        coef_init = np.random.uniform(-coef_limit,coef_limit,(n_features,))
        bias_init = np.zeros((1,))

        gold = SGDRegressor(loss='squared_error',
                            penalty='l2',
                            l1_ratio=0,
                            alpha=alpha,
                            max_iter=max_iter,
                            tol=tol,
                            shuffle=False,
                            learning_rate='constant',
                            eta0=learning_rate,
                            average=False,
                            n_iter_no_change=1)
        gold.fit(X,y,coef_init=coef_init.copy(), intercept_init=bias_init.copy())

        mine = RidgeRegression(alpha=alpha,
                               max_iter=max_iter,
                               learning_rate=learning_rate,
                               coef_init=coef_init.copy(),
                               intercept_init=bias_init.copy(),
                               tol=tol)
        mine.fit(X,y,batch_size=1)

        gold_weights = gold.coef_
        gold_bias = gold.intercept_
        gold_n_iter = gold.n_iter_

        mine_weights = mine.w
        mine_bias = mine.bias
        mine_n_iter = mine.n_iter

        assert_almost_equal(mine_weights, gold_weights, decimal=N_decimal)
        assert_almost_equal(mine_bias, gold_bias, decimal=N_decimal)
        assert_almost_equal(mine_n_iter, gold_n_iter, decimal=N_decimal)
        i+=1

    print('Successfully testing Ridge Regression with Stochastic Gradient Descent!')


def test_lasso_regression_PGD(cases: str) -> None:
    np.random.seed(12346)
    cases = int(cases)
    i = 1
    N_decimal = 4
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        n_features = np.random.randint(5,10)
        X = np.random.rand(n_samples,n_features)

        alpha = np.random.uniform(low=0.01)
        learning_rate = np.random.uniform(low=0.01,high=0.2)

        # make sure that the initial parameters are the same
        coef_limit=1/np.sqrt(n_features)
        coef_init = np.random.uniform(-coef_limit,coef_limit,(n_features,))
        bias_init = np.zeros((1,))

        coef = np.random.uniform(-coef_limit,coef_limit,(n_features,))
        bias = np.random.uniform(-coef_limit,coef_limit,(1,))
        y = X@coef+bias

        gold = SGDRegressor(loss='squared_error',
                            penalty='l1',
                            l1_ratio=1,
                            alpha=alpha,
                            max_iter=max_iter,
                            tol=tol,
                            shuffle=False,
                            learning_rate='constant',
                            eta0=learning_rate,
                            average=False,
                            n_iter_no_change=1)
        gold.fit(X,y,coef_init=coef_init.copy(), intercept_init=bias_init.copy())

        mine = LassoRegression(alpha=alpha,
                               max_iter=max_iter,
                               learning_rate=learning_rate,
                               coef_init=coef_init.copy(),
                               intercept_init=bias_init.copy(),
                               tol=tol)
        mine.fit(X,y,batch_size=1)

        gold_weights = gold.coef_
        gold_bias = gold.intercept_
        gold_n_iter = gold.n_iter_

        mine_weights = mine.w
        mine_bias = mine.bias
        mine_n_iter = mine.n_iter

        assert_almost_equal(mine_weights, gold_weights, decimal=N_decimal)
        assert_almost_equal(mine_bias, gold_bias, decimal=N_decimal)
        assert_almost_equal(mine_n_iter, gold_n_iter, decimal=N_decimal)
        i+=1

    print('Successfully testing Lasso Regression with Proxomal Gradient Descent!')


def test_lasso_regression_CD(cases: str) -> None:

    np.random.seed(123)
    cases = int(cases)
    i = 1
    N_decimal = 3
    max_iter = 100
    tol=1e-3
    while i < cases:
        n_samples = np.random.randint(10, 50)
        n_features = np.random.randint(5,10)
        X = np.random.normal(size=(n_samples,n_features))
        alpha = np.random.uniform(0.1,0.2)

        coef_limit=1/np.sqrt(n_features)
        coef = np.random.uniform(-coef_limit,coef_limit,(n_features,))
        bias = np.random.uniform(-coef_limit,coef_limit,(1,))
        y = X@coef+bias

        gold = Lasso(alpha=alpha,
                     fit_intercept=True,
                     tol=tol,
                     max_iter=max_iter)
        mine = LassoRegressionCD(alpha=alpha,max_iter=max_iter,tol=tol)
        mine.fit(np.copy(X),np.copy(y))
        gold.fit(np.copy(X),np.copy(y))
        gold_weights = gold.coef_
        gold_bias = gold.intercept_
        gold_n_iter = gold.n_iter_
        gold_dual_gap = gold.dual_gap_

        mine_weights = mine.w
        mine_bias = mine.b
        mine_n_iter = mine.this_iter
        mine_dual_gap = mine.dual_gap

        assert_almost_equal(mine_weights, gold_weights, decimal=N_decimal)
        assert_almost_equal(mine_bias, gold_bias, decimal=N_decimal)
        assert_almost_equal(mine_n_iter, gold_n_iter, decimal=N_decimal)
        assert_almost_equal(mine_dual_gap, gold_dual_gap, decimal=N_decimal)

        i+=1

    print('Successfully testing Lasso Regression with Coordinate Descent method!')
