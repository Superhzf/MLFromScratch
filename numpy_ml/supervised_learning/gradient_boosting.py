import numpy as np
from numpy_ml.utils import to_categorical
from ..deep_learning.loss_functions import SquaredLoss, BinomialDeviance
from .decision_tree import RegressionTree

# reference: https://explained.ai/gradient-boosting/

# Why does a GBM model have a learning rate?
# Answer: It reduces contribution of following trees to make the prediction
# more accurate. Suppose the optimal value is 5.23 and at each step, the prediction
# of a tree is 1, then the best value we can landed when learning_rate = 1 is 5 or
# 6. IF learning_rate = 0.1, we can reach 5.2. If learning_rate = 0.01, then
# we can reach 5.23 which is the best one.

# Why does GBM fit gradient instead of residuals?
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
    subsample: float (shoud be in (0,1])
        The fraction of samples to be used for fitting the individual base learners.
    random_state: int
        The random seed for subsample
    """
    def __init__(self,n_estimators,learning_rate,min_samples_split,min_impurity,
                 max_depth,regression,subsample=1,max_features=1,random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state

        assert self.subsample > 0 and self.subsample <= 1
        assert self.max_features > 0 and self.max_features <= 1

        # Square loss for regression, logloss for classification
        if self.regression:
            self.loss = SquareLoss()
        else:
            self.loss = BinomialDeviance()

        # Initialize regression trees
        self.tree_list = []
        for _ in range(self.n_estimators):
            tree = RegressionTree(min_samples_split = self.min_samples_split,
                                  min_impurity = self.min_impurity,
                                  max_depth = self.max_depth,
                                  max_features = self.max_features,
                                  random_state = self.random_state)
            self.tree_list.append(tree)

    def fit(self,X,y):
        if self.subsample < 1:
            np.random.seed(self.random_state)
            index = np.random.choice(len(X),int(self.subsample*len(X)))
            X = X[index]
            y = y[index]
        # initialze predictions using mean value of y
        # y_pred = np.full(np.shape(y),np.mean(y,axis=0))
        y_pred = np.full(np.shape(y),np.mean(y))
        y_pred = np.log(y_pred / (1 - y_pred))
        for i in range(self.n_estimators):
            gradient = self.loss.negative_gradient(y,y_pred)
            # Each tree will fit the gradient instead of residual
            # why negative gradient?
            # https://datascience.stackexchange.com/a/56040
            self.tree_list[i].fit(X,gradient)
            # Update leaf values using line search
            # ref: https://stats.stackexchange.com/questions/330849/how-do-newton-raphson-updates-work-in-gradient-boosting
            self.loss.update_terminal_region(X, y, gradient,self.tree_list[i])
            update = self.tree_list[i].predict(X)
            # Here we subtract negative update
            y_pred += np.multiply(self.learning_rate,update)

    def predict(self,X):
        raise Exception('Not implemented')
        # y_pred = np.array([])
        # for tree in self.tree_list:
        #     update = tree.predict(X)
        #     update = np.multiply(self.learning_rate,update)
        #     y_pred = update if not y_pred.any() else y_pred+update
        #
        # if not self.regression:
        #     # this is softmax
        #     y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        #     # Set label to the value that maximizes probability
        #     y_pred = np.argmax(y_pred, axis=1)
        #
        # return y_pred

    def staged_decision_function(self, X):
        """Compute decision function of X for each iteration."""
        y_pred = np.array([])
        staged_pred = []
        for tree in self.tree_list:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate,update)
            y_pred = update if not y_pred.any() else y_pred+update
            staged_pred.append(y_pred)

        return np.array(staged_pred)


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self,n_estimators=200,learning_rate=0.5,min_samples_split=2,
                 min_var_red = 1e-7,max_depth=4):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_var_red,
            max_depth=max_depth,
            regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, subsample=1,max_features=1, random_state=0):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state)

    def fit(self, X, y):
        # y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)
