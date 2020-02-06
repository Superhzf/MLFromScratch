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
# Answer: softmax is a general form of sigmoid, if we use standard normalization
# it will not give us 0 gradient if it is a correct prediction
class Softmax():
    def __call__(self,x):
        exp_x = np.exp(x-np.max(x,axis=-1,keepdims=True))
        return exp_x/np.sum(exp_x,axis=-1,keepdims=True) # this implementation is more numerically stable

    def gradient(self,x):
        p = self.__call__(x)
        return p*(1-p)

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
        return np.maximum(x,0)

    def gradient(self,x):
        z = np.ones(x.shape)
        z[x<0] = 0
        return x


# Why tanh is better than sigmoid?
# A: Because the absolute value of gradient of Tanh is larger than that of sigmoid
# which makes the network converge faster
