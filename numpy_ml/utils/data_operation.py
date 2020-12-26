import numpy as np
from math import log

def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """
    return np.var(X,axis=0)


def calculate_entropy(y, base=2):
    """
    Calculate the entropy of label array y

    Parameters:
    ------------------------
    y: np.array
        The numpy which is used to calculate ectropy
    base: int/str
        The base of the logrithm, it could be 2 or e
    """
    y = y.flatten().astype(int)
    if len(y) == 0:
        return 1.0
    label_idx = np.unique(y, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)

    if base == 2:
        return -np.sum((pi / pi_sum) * (np.log2(pi) - np.log2(pi_sum)))
    else:
        return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))

class DiscreteSampler():
    def __init__(self, probs, log=False, with_replacement=True):
        """
        Sample from an arbitrary multinomial PMF using Alias Method. Alias
        Method takes O(n) time to initialize, requires O(n) memory, but
        generates samples in constant time.

        tutorial:https://pandasthumb.org/archives/2012/08/lab-notes-the-a.html

        Parameters:
        -----------------
        probs: numpy.array of shape (N,)
            A list of probabilities of the N candidates in the sample space.
            probs[i] returns the probability of i
        log: bool
            Whether the probabilities in probs are in logspace.
        with_replacement: bool
            Whether to generate samples with or without replacement
        """
        if not isinstance(probs, np.ndarray):
            probs=np.array(probs)

        self.log = log
        self.N = len(probs)
        self.probs = probs
        self.with_replacement = with_replacement

        alias = np.zeros(self.N)
        prob = np.zeros(self.N)
        scaled_probs = self.probs + np.log(self.N) if log else self.probs * self.N

        selector = scaled_probs < 0 if log else scaled_probs < 1
        # those candidates whose probability is < 1/N
        small = np.where(selector)[0].tolist()
        # those candidates whose probability is >= 1/N, 1/N is the boundary.
        large = np.where(~selector)[0].tolist()

        while len(small) and len(large):
            l = small.pop()
            g = large.pop()

            alias[l] = g
            prob[l] = scaled_probs[l]

            if log:
                pg = np.log(np.exp(scaled_probs[g]) + np.exp(scaled_probs[l]) - 1)
            else:
                pg = scaled_probs[g] + scaled_probs[l] - 1

            scaled_probs[g] = pg
            to_small = pg < 0 if log else pg < 1
            if to_small:
                small.append(g)
            else:
                large.append(g)

        while len(large):
            prob[large.pop()] = 0 if log else 1

        while len(small):
            prob[small.pop()] = 0 if log else 1

        # prob_table is a dict, the key means the candidate, value is the probability
        # to draw the candidate, 1 - probability is the probability to draw the
        # alias candidate
        self.prob_table = prob
        # alias_table is a dict, the key means shows the candidate, value means
        # that if the probability of key is less than 1, then value is used to
        # fill in the area that is less than 1.
        self.alias_table = alias

    def __call__(self, n_samples=1):
        return self.sample(n_samples)

    def sample(self, n_samples=1):
        """
        Generate random draws from the probs distribution over integers in
        [0, N).

        Parameters:
        -------------------
        n_samples: int
            The number of samples to generate.

        Returns:
        -----------------
        sample: numpy.array of shape (n_samples,)
        """
        ixs = np.random.randint(0, self.N, n_samples)
        p = np.exp(self.prob_table[ixs]) if self.log else self.prob_table[ixs]
        flips = np.random.binomial(1,p)
        samples = [ix if f else self.alias_table[ix] for ix, f in zip(ixs, flips)]

        if not self.with_replacement:
            unique = list(set(samples))
            while len(samples) != len(unique):
                n_new = len(samples) - len(unique)
                samples = unique + self.sample(n_new).tolist()
                unique = list(set(samples))

        return np.array(samples, dtype=int)
