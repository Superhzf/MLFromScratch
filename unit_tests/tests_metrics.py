import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import mutual_info_score
from numpy_ml.metrics.clustering.mutual_information import discrete_mutual_info_score
from numpy.testing import assert_almost_equal

def test_discrete_mutual_info_score(cases: str) -> None:
    seed = 123456
    cases = int(cases)
    i = 1
    N_decimal = 4
    while i < cases:
        random_labels = np.random.RandomState(seed).randint
        n_samples = np.random.randint(1, 100)
        n_classes = np.random.randint(1, 10)
        labels_a = random_labels(low=0, high=n_classes, size=n_samples)
        labels_b = random_labels(low=0, high=n_classes, size=n_samples)
        mine_MI = discrete_mutual_info_score(labels_a, labels_b)
        gold_MI = mutual_info_score(labels_a, labels_b)
        assert_almost_equal(mine_MI, gold_MI, decimal=N_decimal)
        i += 1
