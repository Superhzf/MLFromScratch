import numpy as np

class DecisionNode():
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
