import numpy as np
from numpy_ml.utils import divide_on_feature
from numpy_ml.utils import calculate_variance,calculate_entropy
import random
import warnings
from sklearn.utils import shuffle

# This can work as either an interbal node or a decision node
class Node():
    """
    Class that represents a decision node or leaf in the decision tree. When it
    is a leaf node: value and

    Parameters:
    -----------------------------
    feature_i:int
        Feature index which we want to use as the threshold measure
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction
    value: float
        The class prediction if classification tree, or float value if regression
        tree
    true_branch: Node
        Next decision node for samples where features value meet the threshold
    false_branch: Node
        Next decision node for samples where features value does not meet the
        threshold
    leaf_idx: int
        It is available if it is a leaf node showing.
    """
    def __init__(self,feature_i=None,threshold=None,value=None,
                 true_branch=None,false_branch=None,leaf_idx=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch # left subtree
        self.false_branch = false_branch # right subtree
        self.leaf_idx = leaf_idx

# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """
    Super class of RegressionTree and ClassificationTree.

    Parameters:
    --------------------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree
    min_impurity: float
        The minimum impurity required to split the tree further
    max_depth: int
        The maximum depth of a tree
    loss: function
        Loss function that is used for Gradient Boosting models to calculate
        impurity
    max_features: float (shoud be between (0,1]))
        The % of features to consider when looking for the best split.
    random_state: int
        The random seed for max_features
    """
    def __init__(self,min_samples_split,min_impurity,max_depth=float('inf'),
                 loss=None,max_features = 1,random_state = 0):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        self.loss = loss
        self.max_features = max_features
        self.random_state = random_state
        self._leaf_idx = -1

        assert self.max_features > 0 and self.max_features <= 1

    def fit(self,X,y,loss=None):
        """Build a decision tree"""
        self.root = self._build_tree(X,y)
        self.loss = None

    def _build_tree(self,X,y,current_depth = 0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data

        Parameters:
        --------------------------------
        X: numpy array
            Independent variables
        y: numpy array
            Dependnet variables
        """
        best_impurity = -np.inf
        best_cretiria = None # A dictionary stores feature index and threshold
        best_sets = None # Subsets of the data

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y,axis=1)

        if len(set(y.flatten())) == 1:
            leaf_value = self._leaf_value_calculation(y)
            self._leaf_idx = self._leaf_idx + 1
            return Node(value=leaf_value,leaf_idx=self._leaf_idx)

        # Concatenate x and y
        Xy = np.concatenate((X,y),axis=1)

        # this step will make integer float
        n_samples, n_features = np.shape(X)

        if self.max_features < 1:
            np.random.seed(self.random_state)
            feature_list = np.random.choice(range(n_features),
                                            int(self.max_features*n_features),
                                            replace=False)
            self.random_state = self.random_state + 222
        else:
            np.random.seed(self.random_state)
            feature_list = shuffle(range(n_features),
                                   random_state=self.random_state)

        if n_samples >= self.min_samples_split and current_depth < self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in feature_list:
                # All values of feature_i
                feature_vals = X[:,feature_i]
                levels = np.unique(feature_vals)
                thresholds = (levels[:-1] + levels[1:]) / 2 if len(levels) > 1 else levels

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in thresholds:
                    # Divide X and y according to the feature value, if it is
                    # larger than threshold, then go left, otherwise, go right
                    Xy1,Xy2 = divide_on_feature(Xy,feature_i,threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the target values from the two sets
                        y1 = Xy1[:,n_features:]
                        y2 = Xy2[:,n_features:]

                        # Calculate the impurity
                        impurity = self._impurity_calculation(y,y1,y2)
                        if impurity > best_impurity:
                            best_impurity = impurity
                            best_cretiria = {'feature_i': feature_i,'threshold':threshold}
                            best_sets = {
                                "leftX": Xy1[:,:n_features], # X of left subtree
                                "lefty": Xy1[:,n_features:], # y of left subtree
                                "rightX": Xy2[:,:n_features], # X of right subtree
                                "righty": Xy2[:,n_features:]
                            }

        if best_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets['leftX'],best_sets['lefty'],current_depth+1)
            false_branch = self._build_tree(best_sets['rightX'],best_sets['righty'],current_depth+1)
            # This is an internal node
            # print ("threshold",best_cretiria['threshold'])
            # print ("feature_i", best_cretiria['feature_i'])
            # print ('left samples len', len(best_sets['lefty']))
            # print ('left samples per class', np.bincount(best_sets['lefty'].flatten().astype(int),minlength=3))
            # print ('right samples', len(best_sets['righty']))
            # print ('right samples per class', np.bincount(best_sets['righty'].flatten().astype(int),minlength=3))
            return Node(feature_i = best_cretiria['feature_i'],
                        threshold = best_cretiria['threshold'],
                        true_branch = true_branch,
                        false_branch = false_branch)
        # This is leaf node
        leaf_value = self._leaf_value_calculation(y)
        self._leaf_idx = self._leaf_idx + 1
        return Node(value=leaf_value,leaf_idx=self._leaf_idx)

    def apply(self,x, tree=None):
        """Return leaf idx of the input X"""
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.leaf_idx

        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value,int) or isinstance(feature_value,np.float32)\
            or isinstance(feature_value, float):
            if feature_value>=tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        # Iterate subtree
        return self.apply(x,branch)

    def predict_value(self,x,tree=None, prob = False):
        """
        Do a recursive search down the tree and make a prediction based on the
        value that we end up at
        """
        if tree is None:
            tree = self.root
        # If we are at the leaf node
        if tree.value is not None:
            if isinstance(tree.value, np.ndarray):
                # if np.ndarray means that it is a classification problem
                if prob:
                    return np.mean(tree.value)
                else:
                    return self._majority_vote(tree.value)
            return tree.value
        # Choose the feature that we will iterate
        feature_value = x[tree.feature_i]
        # Determine which branch (left/right) we will follow
        branch = tree.false_branch
        if isinstance(feature_value,int) or isinstance(feature_value,np.float32)\
            or isinstance(feature_value, float):
            if feature_value<=tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        # Iterate subtree
        return self.predict_value(x,branch, prob)

    def predict(self, X):
        """ Make prediction one by one and return the set of labels"""
        y_pred = np.array([self.predict_value(sample, None, False) for sample in X])
        return y_pred

    def predict_prob(self, X):
        """ Make prediction one by one and return the set of  probabilities"""
        pass

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at the leaf node
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

    def _majority_vote(self,y):
        values,counts = np.unique(y,return_counts=True)
        most_freq = values[counts == counts.max()]
        if len(most_freq) > 1:
            warnings.warn("More than 1 class has the same number of \
            frequency, the first class would be used")
            most_freq = most_freq[0]
            return most_freq

        return int(values[counts == counts.max()][0])


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self,y,y1,y2):
        # y: target variable before split
        # y1: target variable at left subtree
        # y2: target variable at right subtree
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1)/len(y)
        frac_2 = len(y2)/len(y)

        # Calculate the variable reduction
        variance_reduction = var_tot - (frac_1*var_1 + frac_2*var_2)
        # sum operation here is to remove the dimension
        return sum(variance_reduction)

    def _mean_of_y(self,y):
        # calculate leaf node value
        value = np.mean(y,axis=0)
        return value if len(value)>1 else value[0]

    def fit(self,X,y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree,self).fit(X,y)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self,y,y1,y2):
        p = len(y1)/len(y)
        entropy_y = calculate_entropy(y)
        entropy_y1 = calculate_entropy(y1)
        entropy_y2 = calculate_entropy(y2)
        p_1 = len(y1)/len(y)

        info_gain = entropy_y-p_1*entropy_y1-(1-p_1)*entropy_y2
        return info_gain

    def _calculate_probability(self, y):
        y = y.flatten().astype(int)
        v = np.bincount(y, minlength=self.n_classes) / len(y)
        return v

    def fit(self,X,y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._calculate_probability
        self.n_classes = len(set(y.flatten()))
        super(ClassificationTree,self).fit(X,y)

# Q: What is boosters in XGBoost?
# A: My understanding is that booster is the type of weak learners. gblinear booster
# makes xgboost pretty much like a lasso regresson. (ref: https://github.com/dmlc/xgboost/issues/332)
# The advantage of gbtree booster is that it can handle non-linearity, it is a huge
# advantage if relations and interactions are unknown. The disadvantage of gbtree
# booster is that it cannnot extrapolate or intertrapolate. If the training data
# has covered the full range of unseen data, then gbdt is a good choice.
# The advantage of gblinear is that it can extrapolate. The disvantage is that
# it assumes a linear relationship. If additional interations can be supplied,
# gblinear would be a power choice.
# ref: https://www.avato-consulting.com/?p=28903&lang=en
# The third booster is DART: it is the dropout version in tree models. The reason
# that DART is introduced is that the first tree of GBDT will have the biggest impact
# to the final prediction and the following trees will only have a small impact,
# more details will be discused below.
# There are two main differences between DART and GBDT boosters.
# The first one is that when building a new tree at iteration t,
# we will have t - 1 trees if it is GBDT boosters, the tth tree will be build
# based on the negative gradient of the previous t - 1 trees. However, DART will only
# consider a random subset of the t - 1 trees which means that the prediction of
# the first tree will be ignored. The second difference is in the
# inference stage. Gbtree just adds up all the trees with a shrinkage factor
# learning rate. However, DART will scale trees by a factor
# ref: https://xgboost.readthedocs.io/en/latest/tutorials/dart.html
# In "DART: Dropouts meet Multiple Additive Regression Trees": "The second place
# at which DART diverges from MART"
class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    """
    def _split(self,y):
        """
        The first half of y is y_true, the second half of y is y_pred
        """
        col = int(np.shape(y)[1]/2)
        y,y_pred = y[:,:col],y[:,col:]
        return y,y_pred

    def _gain(self,y,y_pred):
        # ref: https://xgboost.readthedocs.io/en/latest/tutorials/model.html
        # only the non-zero part is added up
        numerator = np.power((y * self.loss.gradient(y,y_pred)).sum(),2)
        denominator = self.loss.hess(y,y_pred).sum()
        return 0.5*(numerator/denominator)

    def _gain_by_taylor(self,y,y1,y2):
        # Split
        y,y_pred = self._split(y)
        y1,y1_pred = self._split(y1)
        y2,y2_pred = self._split(y2)

        true_gain = self._gain(y1,y1_pred)
        false_gain = self._gain(y2,y2_pred)
        gain = self._gain(y,y_pred)
        return true_gain+false_gain-gain

    def _approximate_update(self,y):
        # y split into y,y_pred
        y,y_pred = self._split(y)
        # Newton's method
        # ref: https://xgboost.readthedocs.io/en/latest/tutorials/model.html
        gradient = np.sum(y*self.loss.gradient(y,y_pred),axis=0)
        hessian = np.sum(self.loss.hess(y,y_pred),axis=0)
        update_approximation = gradient/hessian
        return update_approximation

    def fit(self,X,y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)

# Notes:
# Q: What is the disadvantage of ID3:
# A: 1. It does not handle missing values
#    2. It does not handle numeric input variables
#    3. It only uses entropy and information gain as the impurity function
#    4. It does not perform pruning
# (No max_depth or min_impurity concepts.https://en.wikipedia.org/wiki/ID3_algorithm)
#    5. It can only handle classification problems.
#    6. Only unused features are candidate features.
# (ref for ID3: https://en.wikipedia.org/wiki/ID3_algorithm)
# (ref for entropy and information gain: https://www.saedsayad.com/decision_tree.htm)

# Q: What are other features does ID3 have:
# A: It is not a binary tree. One node may have more than two children

# Q:What is the difference between C4.5 and ID3?
# A: 1. C4.5 can handle continuous variables by creating a threshold and split the
# observations whose attribute values are above the threshold and those are less
# than the threshold
#    2. C4.5 can handle missing attribute values during prediction procedure by
# creating a decision node higher up the tree using the expected value.
# (ref: https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html). If there
# are missing values in building a tree, then we only use observations with known
# values
#    3. C4.5 uses normalized information gain (information gain ratio) to mitigate
# side effects of regular information gain
# (ref: https://www.slideshare.net/marinasantini1/lecture-4-decision-trees-2-entropy-information-gain-gain-ratio-55241087)
#    4. C4.5 will do post-pruning to reduce overfitting by replacing a whole subtree by a leaf node
# the idea is basically considering the replacement of the subtree at each node
# within the tree with a leaf, assigning all observations in that newly assigned leaf
# to the majority class (if a classification problem) or assigning them the mean
# (if a regression problem). If the replacement of this subtree with a leaf leaves
# our overall error/cost no worse, then we keep it, and otherwise we don't.
# We continue iterating over all nodes until the pruning is no longer helpful.

# Q: Other features that C4.5 has:
# A: Like ID3, it is not necessarily a binary tree

# Q: What is the difference between C4.5 and CART?
# A: 1. CART only generates binary trees
#    2. CART handles missing values by using surrogate variables
# (ref: https://www.jamleecute.com/decision-tree-surrogate-in-cart/,
# https://stats.stackexchange.com/questions/171574/meaning-of-surrogate-split).
# Handling missing values is not supported in sklearn
# It works for both training and reference stages.
#    3. The impurity function in CART is Gini index
#    4. CART prunes trees using CCP (cost-complexity pruning) per sklern
