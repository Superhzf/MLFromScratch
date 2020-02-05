import numpy as np
from numpy_ml.utils import divide_on_feature
from numpy_ml.utils import calculate_variance,calculate_entropy

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
    true_branch: Node
        Next decision node for samples where features value meet the threshold
    false_branch: Node
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
        # This is leaf node
        leaf_value = self._leaf_value_calculation(y)
        return Node(value=leaf_value)


    def predict_value(self,x,tree=None):
        """
        Do a recursive search down the tree and make a prediction based on the
        value that we end up at
        """
        if tree is None:
            tree = self.root

        # If we are at the leaf node
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will iterate
        feature_value = x[tree.feature_i]

        # Determine which branch (left/right) we will follow
        branch = tree.false_branch
        if isinstance(feature_value,int) or isinstance(feature_value,float):
            if feature_value>=tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Iterate subtree
        return self.predict_value(x,branch)

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
        self._impurity_calculation = _calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree,self).fit(X,y)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self,y,y1,y2):
        p = len(y1)/len(y)
        entropy_y = calculate_entropy(y)
        entropy_y1 = calculate_entropy(y1)
        entropy_y2 = calculate_entropy(y2)
        p_1 = len(y1)/len(y)

        info_gain = entropy_y-p1*entropy_y1-(1-p1)*entropy_y2
        return info_gain

    def _majority_vote(self,y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurances of samples with label
            count = sum(y==label)
            if count > max_count:
                max_count = count
                most_common = label
        return most_common

    def fit(self,X,y):
        self._impurity_calculation = _calculate_information_gain
        self._leaf_value_calculation = _majority_vote
        super(ClassificationTree,self).fit(X,y)

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
        numerator = np.power(self.loss.gradient(y,y_pred).sum(),2)
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
        gradient = np.sum(self.loss.gradient(y,y_pred),axis=0)
        hess = np.sum(self.loss.hess(y,y_pred),axis=0)
        update_approximation = gradient/hessian
        return update_approximation

    def fit(self,X,y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = _approximate_update
