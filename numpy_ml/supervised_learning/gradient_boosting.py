import numpy as np
from deep_learning.loss_functions import SquareLoss, CrossEntropy
from supervised_learning.decision_tree import RegressionTree

# reference: https://explained.ai/gradient-boosting/

# Why does a GBM model have a learning rate?
# Answer: Learning reduces contribution of following trees to make the prediction
# more accurate. Suppose the optimal value is 5.23 and at each step, the prediction
# of a tree is 1, then the best value we can land when learning_rate = 1 is 5 or
# 6. IF learning_rate = 0.1, we can reach 5.2. If learning_rate = 0.01, then
# we can reach 5.23 which is the best one.

# Why does GBM fit gradient?
# Answer: The idea of boosing is that the following models will correct mistakes
# made by the previous weak learners. Let's say if the loss function is MSE, then
# after t-1 iterations, we have t-1 models, for the t-th model, obj_t = [y-(y_(t-1)+f_t)]^2
# obj_t = constant + [2f_t*(y_(t-1)-y)+f_t^2], so now the question becomes what
# should be f_t so that obj_t is minimized, it is a quadratic formula, f_t= y - y_(t-1)
# which is the gradient

# How to understand that predictions from following trees are subtracted from
# the previous predictions?
# A: Because we want the most significant improvement on the prediction
class GradientBoosting(object):
    """
    Super class of GradientBoostingClassifier and GradientBoostingRegressor. Use
    a collection of regression trees that trains on predicting the gradient of
    the loss function

    Parameters
    ---------------------------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient
        during training
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree
    min_impurity: float
        The minimum impurity required to further split the tree
    max_depth: int
        The maximum depth of a tree
    regression: boolean
        True if it is a regression problem otherwise False
    """
    def __init__(self,n_estimators,learning_rate,min_samples_split,min_impurity,
                 max_depth,regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # Square loss for regression, logloss for classification
        if self.regression:
            self.loss = SquareLoss()
        else:
            self.loss = CrossEntropy()

        # Initialize regression trees
        self.trees = []
        for _ in range(self.n_estimators):
            tree = RegressionTree(min_samples_split = self.min_samples_split,
                                  min_impurity = self.min_impurity,
                                  max_depth = self.max_depth)
            self.trees.append(tree)

    def fit(self,X,y):
        # initialze predictions using mean value of y
        y_pred = np.full(np.shape(y),np.mean(y,axis=0))
        for i in range(self.n_estimators):
            # for squared loss, the gradient is residual
            gradient = self.loss.gradient(y,y_pred)
            # Each tree will fit the gradient instead of residual
            self.tree[i].fit(X,gradient)
            update = self.trees[i].predict(X)
            # Here we subtract negative update
            y_pred -= np.multiply(self.learning_rate,update)

    def predict(self,X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate,update)
            y_pred = -update if not y_pred.any() else y_pred-update

        if not self.regression:
            # Turn probability distribution
            y_pred = np.exp(y_pred)/np.expand_dims(np.sum(np.exp(y_pred),axis=1),axis=1) # this is softmax
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred

class GradientBoostingRegressor(GradientBoosting):
    def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
                 min_var_red = 1e-7,max_depth=4,debug=False):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)
