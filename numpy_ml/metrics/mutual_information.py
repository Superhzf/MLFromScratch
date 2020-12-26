import numpy as np
from scipy.special import comb
from ..utils import calculate_entropy

def discrete_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
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
    assert len(labels_a) == len(labels_b), \
        "The length of label_a and that of label_b is supposed to be the same"
    unique_a = set(labels_a)
    unique_b = set(labels_b)
    N = len(labels_a)
    MI = 0
    for this_a in unique_a:
        for this_b in unique_b:
            dist_a = labels_a == this_a
            dist_b = labels_b == this_b
            jnt_dist_ab = np.sum(dist_a & dist_b)/N
            if jnt_dist_ab == 0:
                continue
            dist_a = dist_a.sum()/N
            dist_b = dist_b.sum()/N
            MI += jnt_dist_ab*np.log(jnt_dist_ab/(dist_a*dist_b))
    return MI


def adjusted_discrete_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Adjusted mutual information between two clusterings. The motivation is
    accounting for the fact that the regular MI is generally higher for two
    clusterings with a larger number of clusters, regardless of whether there
    is actually more information shared.

    Ref: https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf

    Parameters:
    ------------------
    labels_a: np.ndarray
        One of the two discrete distributions.
    labels_b: np.ndarray
        One of the two discrete distributions. The order of labels_a and labels_b
        does not matter.

    Output:
    ------------------
    AMI: float
        The adjusted mutual information of two discrete distributions.
    """
    assert len(labels_a) == len(labels_b), \
        "The length of label_a and that of label_b is supposed to be the same"
    # calculate the partial sums of the contingency table
    # Refer to https://en.wikipedia.org/wiki/Adjusted_mutual_information for more details
    def calcul_marginal(labels_outer, unique_outer, labels_inner, unique_inner):
        output = []
        for this_outer in unique_outer:
            margin_a = labels_outer == this_outer
            margin_ab = 0
            for this_inner in unique_inner:
                margin_b = labels_inner == this_inner
                margin_ab += (margin_a & margin_b).sum()
            output.append(margin_ab)
        return np.array(output)

    unique_a = np.array(sorted(list(set(labels_a))))
    unique_b = np.array(sorted(list(set(labels_b))))

    if len(unique_a) == len(unique_b) and len(unique_a)==1:
        return 1.0

    C = len(unique_a)
    R = len(unique_b)

    a_i = calcul_marginal(labels_a, unique_a, labels_b, unique_b)
    b_j = calcul_marginal(labels_b, unique_b, labels_a, unique_a)

    N = len(labels_a)

    expected_mi = 0
    for this_a in range(len(unique_a)):
        for this_b in range(len(unique_b)):
            min_n_ij = max(1, a_i[this_a]+b_j[this_b]-N)
            max_n_ij = min(a_i[this_a], b_j[this_b])+1
            for this_n_ij in range(min_n_ij, max_n_ij):

                term1 = this_n_ij / N
                term2 = np.log(N*this_n_ij/(a_i[this_a]*b_j[this_b]))
                numer = comb(a_i[this_a], this_n_ij) * comb(N-a_i[this_a], b_j[this_b]-this_n_ij)
                denomi = comb(N, b_j[this_b])
                term3 = numer/denomi
                expected_mi += (term1 * term2 *term3)

    mi = discrete_mutual_info(labels_a, labels_b)
    # The natural base comes from the implementation by sklearn
    entropy_a = calculate_entropy(labels_a, 'e')
    entropy_b = calculate_entropy(labels_b, 'e')
    avg_entropy = np.mean([entropy_a, entropy_b])
    denominator = avg_entropy - expected_mi
    if denominator < 0:
        denominator = min(denominator, -np.finfo('float64').eps)
    else:
        denominator = max(denominator, np.finfo('float64').eps)
    ami = (mi - expected_mi)/(denominator)
    return ami
