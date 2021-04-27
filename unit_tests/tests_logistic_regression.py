# import sys
# sys.path.append('..')
# import numpy as np
# from numpy.testing import assert_almost_equal
# from scipy.stats import logistic
# from sklearn.linear_model import LogisticRegression
# from numpy_ml.supervised_learning.logistic_regression import LogisticRegression_LBFGS
#
# def test_logistic_regression_lbfgs(cases: str) -> None:
# #     np.random.seed(12346)
#     np.random.seed(123)
#     cases = int(cases)
#     i = 1
#     N_decimal = 4
#     max_iter = 100
#     tol=1e-3
#     while i < cases:
#         n_samples = np.random.randint(10, 50)
#         n_features = np.random.randint(5,10)
#         X = np.random.rand(n_samples,n_features)
#
#         # make sure that the initial parameters are the same
#         coef_limit=1/np.sqrt(n_features)
# #         coef_init = np.random.uniform(-coef_limit,coef_limit,(n_features,))
# #         bias_init = np.zeros((1,))
#
#         coef = np.random.uniform(-coef_limit,coef_limit,(n_features,))
#         bias = np.random.uniform(-coef_limit,coef_limit,(1,))
#         prob = logistic.cdf(X@coef+bias)
#         prob_mean = np.mean(prob)
#         y = 1*(prob > prob_mean)
#
# #         extra_col = np.ones((n_samples,1))
# #         X2 = np.append(X, extra_col, axis=1)
# #         coef2 = np.concatenate([coef,bias])
# #         prob2 = logistic.cdf(X2@coef2)
# #         prob1 = logistic.cdf(X@coef+bias)
# #         print (prob1-prob2)
# #         break
#
#         gold = LogisticRegression(penalty='none',
#                                   fit_intercept=True,
#                                   tol=tol,
#                                   solver='lbfgs',
#                                   max_iter=1)
#         mine = LogisticRegression_LBFGS(max_iter=1,gtol=tol)
#         gold.fit(X,y)
#         mine.fit(X,y)
#         gold_weights = gold.coef_
#         gold_bias = gold.intercept_
#         gold_n_iter = gold.n_iter_
#         print ('gold_weights',gold_weights)
#         print ('gold_bias',gold_bias)
#         print ('gold_n_iter',gold_n_iter)
#
#         mine_weights = mine.w
#         mine_bias = mine.b
#         mine_n_iter = mine.this_iter
#         print ('mine_weights',mine_weights)
#         print ('mine_bias',mine_bias)
#         print ('mine_n_iter',mine_n_iter)
#
# #         assert_almost_equal(mine_weights, gold_weights, decimal=N_decimal)
# #         assert_almost_equal(mine_bias, gold_bias, decimal=N_decimal)
# #         assert_almost_equal(mine_n_iter, gold_n_iter, decimal=N_decimal)
# #         print ('passed')
#         break
#         i+=1
#
# #     print('Successfully testing logistic regression with newton method!')
#
