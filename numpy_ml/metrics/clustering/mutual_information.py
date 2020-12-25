import numpy as np

def discrete_mutual_info_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    To calculate the mutual information between discrete distributions. It is
    used to measure how much knowing one of the two variables reduces uncertainty
    about the other. The intuition is how many bits needed to store X givend that
    we already know the value of Y.

    Ref: https://en.wikipedia.org/wiki/Mutual_information

    Parameters:
    ------------------
    labels_a: np.ndarray
        One of the two discrete distributions.
    labels_b: np.ndarray
        One of the two discrete distributions. The order of labels_a and labels_b
        does not matter.

    Output:
    ------------------
    MI: float
        The mutual information of two discrete distributions.
    """
    assert len(labels_a) == len(labels_b), "The length of label_a and that of label_b is supposed to be the same"
    unique_a = set(labels_a)
    unique_b = set(labels_b)
    N = len(labels_a)
    MI = 0
    for this_a in unique_a:
        for this_b in unique_b:
            dist_a = labels_a == this_a
            dist_b = labels_b == this_b
            jnt_dist_ab = np.sum(dist_a & dist_b)*1.0/N
            if jnt_dist_ab == 0:
                continue
            dist_a = dist_a.sum()*1.0/N
            dist_b = dist_b.sum()*1.0/N
            MI += jnt_dist_ab*np.log(jnt_dist_ab/(dist_a*dist_b))
    return MI
