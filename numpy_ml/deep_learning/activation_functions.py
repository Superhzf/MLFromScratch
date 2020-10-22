import numpy as np

# Why sigmoid function instead of anything else?
# reference: https://stats.stackexchange.com/questions/162988/why-sigmoid-function-instead-of-anything-else/318209#318209
# Answer: The function should fullfil several attributes,
# 1. The range of the function should be between (0,1)
# 2. The function should be monotonous. continuous and differentiable
# 3. f(0) = 0.5
# 4. It should be symmetrical which means f(-x) = 1-f(x)
# 5. We need a strong gradient and 0 cost when the prediction is wrong and a
# small gradient and large |cost| when the prediction is correct
# For sigmoid function, we will proof how it meets crietia 5.
# Let's if y_true = 1, z is the output of the last non-avtivation layer, after
# sigmoid, P = 1/(1+e^(-z)), loss = -log(P) = -log(e^z/(1+e^z)) = -z+log(1+e^z)
# If z is large which means the prediction is correct, then loss = -z + z = 0
# and dz = 0 which is perfect
# In the other hand, if z is small (|z| is large) which means the prediction is wrong,
# then loss = -z which is desirble,so dz = -1 which means we can move close to
# the optimal value. Problem solved
class Sigmoid():
    def __call__(self,x):
        return 1/(1+np.exp(-x))

    def gradient(self,x):
        p = self.__call__(x)
        return p*(1-p)

# Why Softmax instead of standard normalization ?
# Answer: softmax play nicely with logloss. The gradient is easier to calculate and
# numerically stable.
# ref: https://stats.stackexchange.com/questions/289369/log-probabilities-in-reference-to-softmax-classifier

# Q: Why the regular definition of softmax is numerically unstable?
# A: When the output of the layer right before the last one is extremely large
# or extremely small (absolute value), exp(output) would be close to inf or 1
# which is useless. For example, when the output is [1/10000, 1/20000, 1/30000]
# or [10000, 20000, 30000], the result is close to inf and 1.
class Softmax():
    """
    This softmax function has to be used together with cross entropy loss for the
    convenience of the gradients calculation.
    """
    def __call__(self,x):
        exp_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=-1,keepdims=True) # this implementation is more numerically stable

    def gradient(self,x):
        p = self.__call__(x)
        # we do not do any gradieny calculaion here because we assume that
        # softmax will always be used with cross entropy loss together
        # at the same calculating the gradient of cross entropy loss w.r.t the
        # input of softmax layer is much easier.
        return np.ones_like(p)

class FullSoftmax():
    """
    This softmax function can be used with anything else, the inspiration comes
    from the transformer architecture.

    ref: https://stats.stackexchange.com/questions/267576/matrix-representation-of-softmax-derivatives-in-backpropagation
    """
    def __call__(self,x):
        """
        Parameters:
        ---------------
        x: numpy.array of shape (n_ex, n_classes)
            The input of softmax layer, usually the input is the prediction for
            n_classes classes.
        """
        exp_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=-1,keepdims=True) # this implementation is more numerically stable

    def gradient(self, x, dLdoutput):
        dX = []
        p=self.__call__(x)
        for this_obs, this_dLdoutput in zip(p, dLdoutput):
            # set up the shape of this_obs from (n_classes,) to be (1, n_classes)
            this_obs = this_obs[None,...]
            diag_value = this_obs*(1-this_obs)
            diag = np.diagflat(diag_value)
            off_diag = -1*(this_obs.transpose() @ this_obs)
            np.fill_diagonal(off_diag,0)
            dXi = this_dLdoutput[None,...] @ (diag + off_diag)
            dX.append(dXi)
        return np.vstack(dX)



# What is a dead ReLU problem? Why does it happen?
# A: Dead ReLU means that the activations are the same (0 as it happens) and
# it will never recover because the gradient of 0 is always 0, it means it takes
# no role in discriminating inputs. Probably this is arrived by learning a
# large negative bias term for its weights. Besides, if the learning rate is large
# then the updated weights could be less than 0 and dead.

# ReLU can help gradients vanishing problem in regular deep neural networks caused
# by sigmoid/tanh
class ReLU():
    def __call__(self,x):
        # return np.maximum(x,0)
        return np.clip(x, 0, np.inf)

    def gradient(self,x):
        # The way of implementation is more numerically stable but I don't
        # know why
        return (x > 0).astype(int)
        # z = np.ones(x.shape)
        # z[x<0] = 0
        # return x


# Why tanh is better than sigmoid?
# A: Because the absolute value of gradient of Tanh is larger than that of sigmoid
# which makes the network converge faster. Somebody says we prefer zero averages,
# I don't know why, batch normalization has an average alpha instead of zero.

class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)
