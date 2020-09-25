import numpy as np

class CosineAnnealingLR():
    def __init__(self, min_lr, t_max):
        self.min_lr = min_lr
        self.t_max = t_max

    def __call__(self, curr_epoch):
        return self.get_learning_rate(curr_epoch)

    def get_max_lr(self, max_lr):
        self.max_lr = max_lr

    def get_learning_rate(self, curr_epoch):
        curr_lr = self.min_lr+1/2*(self.max_lr-self.min_lr)*(1+np.cos(np.pi*curr_epoch/self.t_max))
        return curr_lr
