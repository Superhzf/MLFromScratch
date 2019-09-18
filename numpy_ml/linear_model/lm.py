import numpy as np
from sklearn.datasets import load_iris


def sigmoid(z):
    neg_z = np.negative(z)
    a = 1/(1+np.exp(neg_z))
    return a


def logloss(y_true,y_hat):
    loss = np.mean(-1*(y_true*np.log(y_hat)+(1+np.negative(y_true))*np.log(1+np.negative(y_hat))))
    return loss


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.loss = []
        self.training_loss = None

    def fit(self, X, Y, alpha, round):
        # n features, m samples
        n, m = X.shape
        self.b = np.random.rand(1,1)
        self.W = np.random.rand(n,1)
        for i in range(round):
            Z = np.matmul(np.transpose(self.W),X)+self.b
            A = sigmoid(Z)
            dZ = A-Y
            dW = np.matmul(X,np.transpose(dZ))/m
            db = np.sum(dZ)/m
            self.W = self.W - np.multiply(dW, alpha)
            self.b = self.b - np.multiply(db, alpha)
            round_n_loss = logloss(Y, A)
            self.loss.append(round_n_loss)
        self.training_loss = round_n_loss

data = load_iris()
X = np.transpose(data['data'])
Y = np.reshape(data['target'],(1,-1))

lr = LogisticRegression()
lr.fit(X,Y,0.01,100)

print (lr.training_loss)