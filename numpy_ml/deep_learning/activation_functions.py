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
# extremely small (absolute value), exp(output) would be inf or 1/(len(output))
# which is useless. For example, when the output is [1/10000, 1/20000, 1/30000]
# or [10000, 20000, 30000]
class Softmax():
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

# What is a dead ReLU problem? Why does it happen?
# A: Dead ReLU means that the activations are the same (0 as it happens) and
# it will never recover because the gradient of 0 is always 0, it means it takes
# no role in discriminating inputs. Probably this is arrived at by learning a
# large negative bias term for its weights. Besides, if the learning rate is large
# then the updated weights could less than 0 and dead.

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
