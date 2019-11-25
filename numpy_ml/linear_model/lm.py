import numpy as np
import pandas as pd


def sigmoid(z):
    a = 1/(1+np.exp(-1*z))
    return a


def logloss(y_true,y_hat):
    loss = -1*np.sum(y_true*np.log(y_hat) +(1-y_true)*np.log(1-y_hat))
    return loss


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, x, y):
        m, n = x.shape
        self.b = np.random.rand(1,m)
        self.W = np.random.rand(1,n)
        z = np.dot(self.W,np.transpose(x))
        a = sigmoid(z)
        loss = logloss(y,a)
        dz = a - y
        dw = (1/m)*x


train = pd.DataFrame({"var1":[1,2,3],'var2':[4,5,6],'y':[1,0,1]})
lr = LogisticRegression()
lr.fit(train[['var1','var2']].values,train['y'].values)
print (lr.W,lr.b)
print (np.shape(lr.W))