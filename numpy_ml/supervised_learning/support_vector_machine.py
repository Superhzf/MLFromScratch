from __future__ import division, print_function
import numpy as np
import random

# ref: http://web.cs.iastate.edu/~honavar/smo-svm.pdf
# The idea of SMO:
# 1. Optimize two variables (alpha1 and alpha2) at a time, the reason for optimizing
# two variables is that sum of alpha_i * y_i == 0
# 2. There is a closed form updates which makes the update really fast
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

    def fit(self, X, y):
        """
        X: numpy.array
            Observations
        y: numpy.array
            The ground truth
        """
        n = X.shape[0]
        d = X.shape[1]
        alpha = np.zeros(n)
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(n):
                # randomly select an integer between 0 and n-1 (inclusive)
                # and it is not equalt to j
                i = self.get_random_int(0, n-1, j)
                x_i = X[i, :]
                x_j = X[j, :]
                y_i = y[i]
                y_j = y[j]
                # k_ij = kernel(x_i, x_i) + kernel(x_j, x_j)  - 2 * kernel(x_i, x_j)
                k_ij = 2 * kernel(x_i, x_j) - kernel(x_i, x_i) - kernel(x_j, x_j)
                alpha_prime_j = alpha[j]
                alpha_prime_i = alpha[i]

                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j (the difference between the prediction and
                # the ground truth)
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                L, H = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                if k_ij < 0:
                    alpha[j] = alpha_prime_j - y_j * (E_i - E_j) / k_ij
                    if alpha[j] >= H:
                        alpha[j] = H
                    elif alpha[j] <= L:
                        alpha[j] = L
                else: # k_ij == 0
                    alpha[j] = L
                    w_L = self.calc_w(alpha, y, X)
                    b_L = self.calc_b(X, y, w_L)
                    Lobj = np.dot(w_L.T, X.T)+b_L
                    alpha[j] = H
                    w_H = self.calc_w(alpha, y, X)
                    b_H = self.calc_b(X, y, w_L)
                    Hobj = p.dot(w_H.T, X.T)+b_H
                    if Lobj > Hobj:
                        alpha[j] = L
                    elif Lobj < Hobj:
                        alpha[j] = H
                    else:
                        alpha[j] = alpha_prime_j

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha_prev-alpha)
            print (diff)
            if diff < self.epsilon:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of {} iterations".format(self.max_iter))
                return

        # Compute the final model parameters
        self.w = self.calc_w(alpha, y, X)
        self.b = self.calc_b(X, y, self.w)

        # Get support vectors
        alpha_idx = np.where(alpha != 0)[0]
        self.support_vectors_ = X[alpha_idx, :]

    def decision_function(self, X):
        return np.dot(self.w.T, X.T)+self.b

    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def get_random_int(self, a, b , z):
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = random.randint(a,b)
            cnt += 1
        return i

    def compute_L_H(self, C, alpha_prime_j,alpha_prime_i, y_j, y_i):
        # Calculate left bound and right bound of alpha
        if y_i != y_j:
            return max(0,alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j)
        else:
            return max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha, y))

    def calc_b(self, X, y ,w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    # Prediction
    def predict(self, X, w, b):
        return np.sign(np.dot(w.T, X.T)+b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.predict(x_k, w, b) - y_k
