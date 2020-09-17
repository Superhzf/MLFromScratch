import numpy as np

# ref: http://web.cs.iastate.edu/~honavar/smo-svm.pdf

# Q: what is the difference between SVM and logistic regression?
# A: 1. LR optimizes the likelihood. SVM is much more geometrically motivated.
# Instead of assuming a probabilistic model, we're trying to find a particular
# optimal separating hyperplane, where we define "optimality" in the context of
# the support vectors.
#   2. Logistic regression is more explainable. There are statistical tools
# available to understand the relationship between predictors and the target.
# and statistically test the relationship between the predictors and the target.
# However, logistic regression has some assumptions: observations are independent
# to each other; little or no multicollinearity among the independent variables;
# linear relationship between log odds and independnet variables.
#
class SVM():
    """
    An implementation of SVM algorithm using the Sequential Minimal Optimization
    (SMO) algorithm for training
    """
    def __init__(self, max_iter=1000,kernel_type='linear', C=1.0, epsilon=0.001):
        self.kernels={
            'linear': self.kernel_linear
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.b = 0.0

    def _take_one_step(self, i, j, kernel, alpha):
        if i == j:
            return 0

        E_i = self.errors[i]
        E_j = self.errors[j]

        x_i = self.X[i, :]
        x_j = self.X[j, :]
        y_i = self.y[i]
        y_j = self.y[j]
        k_ij = kernel(x_i, x_j)
        k_ii = kernel(x_i, x_i)
        k_jj = kernel(x_j, x_j)

        r_j = E_j * y_j
        if (r_j < -self.epsilon and alpha[j] < self.C) or (r_j > self.epsilon and alpha[j] > 0):
            pass
        else:
            return 0

        eta = 2 * k_ij - k_ii - k_jj
        alpha_prime_j = alpha[j]
        alpha_prime_i = alpha[i]

        L, H = self._compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)
        if L == H:
            return 0

        if eta < 0:
            alpha[j] = alpha_prime_j - y_j * (E_i - E_j) / eta
            if alpha[j] > H:
                alpha[j] = H
            elif alpha[j] < L:
                alpha[j] = L
        else: # eta == 0
            print ('!!!!eta == 0!!!!')
            alpha[j] = L
            w_L = self._calc_w(alpha, self.y, self.X)
            # b_L = self.calc_b(self.X, self.y, w_L)
            Lobj = np.dot(w_L.T, X.T)-self.b
            alpha[j] = H
            w_H = self._calc_w(alpha, self.y, self.X)
            # b_H = self.calc_b(self.X, self.y, w_L)
            Hobj = p.dot(w_H.T, X.T)-self.b
            if Lobj > Hobj:
                alpha[j] = L
            elif Lobj < Hobj:
                alpha[j] = H
            else:
                alpha[j] = alpha_prime_j

        # Push a2 to 0 or C if very close
        if alpha[j] < 1e-8:
            alpha[j] = 0.0
        elif alpha[j] > (self.C - 1e-8):
            alpha[j] = self.C

        if np.abs(alpha[j] - alpha_prime_j) < self.epsilon:
            alpha[j] = alpha_prime_j
            return 0

        alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

        # Update the bias item
        b1 = E_i + y_i * (alpha[i] - alpha_prime_i) * k_ii + y_j * (alpha[j] - alpha_prime_j) * k_ij + self.b
        b2 = E_j + y_i * (alpha[i] - alpha_prime_i) * k_ij + y_j * (alpha[j] - alpha_prime_j) * k_jj + self.b
        if alpha[i] > 0 and alpha[i] < self.C:
            new_b = b1
        elif alpha[j] > 0 and alpha[j] < self.C:
            new_b = b2
        else:
            new_b = (b1+b2)/2

        if alpha[i] > 0 and alpha[i] < self.C:
            self.errors[i] = 0
        if alpha[j] > 0 and alpha[j] < self.C:
            self.errors[j] = 0

        non_opt = [this_sample for this_sample in range(self.n) if (this_sample != i and this_sample != j)]
        self.errors[non_opt] = self.errors[non_opt] +\
                                y_i * (alpha[i] - alpha_prime_i) * kernel(x_i, self.X[non_opt]) +\
                                y_j * (alpha[j] - alpha_prime_j) * kernel(x_j, self.X[non_opt]) + self.b - new_b
        self.b = new_b
        return 1

    def fit(self, X, y):
        """
        X: numpy.array
            Observations
        y: numpy.array
            The ground truth
        """
        self.X = np.copy(X)
        self.y = np.copy(y)
        if len(np.unique(y)) != 2:
            raise Exception("The number of classes is not understood!")
        self.transformed = False
        # Adjust the target variabe to make it become either -1 or 1
        if np.unique(self.y)[0] != -1 or np.unique(self.y)[1] != 1:
            self.transformed = True
            self.y_unique = np.unique(y)
            self.y[self.y==self.y_unique[0]] = -1
            self.y[self.y==self.y_unique[1]] = 1

        self.n = self.X.shape[0]
        d = self.X.shape[1]
        alpha = np.zeros(self.n)
        kernel = self.kernels[self.kernel_type]
        count = 0
        # Initialize errors
        self.w = self._calc_w(alpha, self.y, self.X)
        self.errors = self._predict(self.X, self.w, self.b) - self.y
        examine_all = True
        num_changed = 0
        while num_changed > 0 or examine_all:
            num_changed = 0
            count += 1
            if examine_all:
                samples = np.array(range(self.n))
            else:
                samples = np.where((alpha != 0) & (alpha != self.C))[0]
            for j in samples:

                if len(alpha[(alpha != 0) & (alpha != self.C)]) > 1:
                    if self.errors[j] > 0:
                        i = np.argmin(self.errors)
                    else:
                        i = np.argmax(self.errors)
                    this_change = self._take_one_step(i, j, kernel, alpha)
                    if this_change > 0:
                        num_changed += this_change
                        continue
                # Loop through non-zero and non-C alphas
                i_candidate_list = np.where((alpha != 0) & (alpha != self.C))[0]
                for i in i_candidate_list:
                    this_change = self._take_one_step(i, j, kernel, alpha)
                    if this_change > 0:
                        num_changed += this_change
                        continue

                # loop through all alphas
                for i in range(self.n):
                    this_change = self._take_one_step(i, j, kernel, alpha)
                    if this_change > 0:
                        num_changed += this_change
                        continue

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            if count >= self.max_iter:
                print("Iteration number exceeded the max of {} iterations".format(self.max_iter))
                return

        # Compute the final model parameters
        self.w = self._calc_w(alpha, self.y, self.X)

        # Get support vectors
        alpha_idx = np.where(alpha != 0)[0]
        self.support_vectors_ = self.X[alpha_idx, :]

    def decision_function(self, X):
        return np.dot(self.w.T, X.T) - self.b

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def _compute_L_H(self, C, alpha_prime_j,alpha_prime_i, y_j, y_i):
        # Calculate left bound and right bound of alpha
        if y_i != y_j:
            return max(0,alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j)
        else:
            return max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j)

    def _calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    # Prediction
    def _predict(self, X, w, b):
        # return np.sign(np.dot(w.T, X.T)+b).astype(int)
        return np.dot(w.T, X.T)-b

    def predict(self, X):
        res = np.sign(np.dot(self.w.T, X.T)+self.b).astype(int)
        if self.transformed:
            res[res == -1] = self.y_unique[0]
            res[res == 1] = self.y_unique[1]
        return res
