# I cannot do unit test for tree-based algorithms for now because when there
# are more than one best splits, the DecisionTree algorithms from sklearn
# cannot return a deterministic variable.
# For more details, please refer to this PR:
# https://github.com/scikit-learn/scikit-learn/pull/12364
# and this issue:
# https://github.com/scikit-learn/scikit-learn/issues/12259
# once this PR is ready, unit test can be done in a deterministic way. ^^

# def test_DecisionTreeClassifier(N=1):
#     i = 1
#     np.random.seed(12345)
#     while i <= N:
#         n_ex = np.random.randint(2, 100)
#         n_feats = np.random.randint(2, 100)
#         max_depth = np.random.randint(1, 5)
#
#         print ("n_ex", n_ex, "n_feats", n_feats, "max_depth", max_depth)
#         # create classification problem
#         n_classes = np.random.randint(2, 10)
#         X, Y = make_blobs(
#             n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=i
#         )
#
#         # initialize models
#         mine = ClassificationTree(min_samples_split=2,
#                                   min_impurity=0,
#                                   max_depth=max_depth,
#                                   max_features=1,
#                                   random_state=0)
#         gold = DecisionTreeClassifier(
#             criterion='entropy',
#             splitter="best",
#             max_depth=max_depth,
#             min_samples_split=2,
#             min_samples_leaf=1,
#             min_weight_fraction_leaf=0,
#             max_features=None,
#             max_leaf_nodes=None,
#             min_impurity_decrease=0,
#             random_state=0,
#         )
#         # fit model
#         mine.fit(X, Y)
#         gold.fit(X, Y)
#         display(graphviz.Source(export_graphviz(gold)))
#
#
#         children_left = gold.tree_.children_left
#         children_right = gold.tree_.children_right
#         feature = gold.tree_.feature
#         threshold = gold.tree_.threshold
#         values = gold.tree_.value
#
#         print ("children_left", children_left)
#         print ("children_right", children_right)
#         print ("feature", feature)
#         print ("threshold", threshold)
#         print ("values", values)
#
#         assert compare_trees(mine, gold) is True
#
#         i += 1
