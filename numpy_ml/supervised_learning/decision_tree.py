import numpy as np
from numpy_ml.utils import divide_on_feature

# This can work as either an interbal node or a decision node
class Node():
    """
    Class that represents a decision node or leaf in the decision tree

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
    true_branch: DecisionNode
        Next decision node for samples where features value meet the threshold
    false_branch: DecisionNode
        Next decision node for samples where features value does not meet the
        threshold
    """
    def __init__(self,feature_i,threshold,value=None,true_branch,false_branch):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch # left subtree
        self.false_branch = false_branch # right subtree

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
    """
    del __init__(self,min_samples_split,min_impurity,max_depth=float('inf'),loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        self.loss = loss

    del fit(self,X,y,loss=None):
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
        largest_impurity = 0
        best_cretiria = None # A dictionary stores feature index and threshold
        best_sets = None # Subsets of the data

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y,axis=1)

        # Concatenate x and y
        Xy = np.concat((X,y),axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:,feature_i],axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y according to the feature value, if it is
                    # larger than threshold, then go left, otherwise, go right
                    Xy1,Xy2 = divide_on_feature(Xy,feature_i,threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the target values from the two sets
                        y1 = Xy1[:,n_features:]
                        y2 = Xy2[:,n_features:]

                        # Calculate the impurity
                        impurity = self._impurity_calculation(y,y1,y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_cretiria = {'feature_i': feature_i,'threshold':threshold}
                            best_sets = {
                                "leftX": Xy1[:,:n_features], # X of left subtree
                                "lefty": Xy1[:,n_features:], # y of left subtree
                                "rightX": Xy2[:,:n_features], # X of right subtree
                                "lefty": Xy2[:,n_features:]
                            }
        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets['leftX'],best_sets['lefty'],current_depth+1)
            false_branch = self._build_tree(best_sets['rightX'],best_sets['righty'],current_depth+1)
            # This is an internal node
            return Node(feature_i = best_cretiria['feature_i'],
                        threshold = best_cretiria['threshold'],
                        true_branch = true_branch,
                        false_branch = false_branch)
        # This is an internal node
        leaf_value = self._leaf_value_calculation(y)
        return Node(value=leaf_value)
