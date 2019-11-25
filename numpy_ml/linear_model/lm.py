import numpy as np
from sklearn.datasets import load_breast_cancer


def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a


def logloss(y_true,y_hat):
    loss = np.mean(-1*(y_true*np.log(y_hat+0.0001)+(1-y_true)*np.log(1-y_hat+0.0001)))
    return loss


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.loss = []
        self.training_loss = None

    def fit(self, X, Y, alpha, round, normalize = False):
        # n features, m samples
        n, m = X.shape
        self.b = np.random.rand(1, 1)
        self.W = np.random.rand(n, 1)
        if normalize:
            mean = np.mean(X,axis=1)
            std = np.std(X,axis=1)
            for field in range(n):
                X[field] = (X[field] - mean[field])/std[field]
        for i in range(round):
            Z = np.matmul(np.transpose(self.W),X)+self.b
            A = sigmoid(Z)
            dZ = A-Y
            dW = np.matmul(X, np.transpose(dZ))/m
            db = np.sum(dZ)/m
            self.W = self.W - dW*alpha
            self.b = self.b - db*alpha
            round_n_loss = logloss(Y, A)
            self.loss.append(round_n_loss)
        self.training_loss = round_n_loss


data = load_breast_cancer()
X = np.transpose(data['data'])
Y = np.reshape(data['target'],(1,-1))

<<<<<<< HEAD
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
=======
>>>>>>> a2fb90defbf89f74f79c617758caa63a3457cbee
lr = LogisticRegression()
lr.fit(X,Y,0.01,100)

print (lr.training_loss)