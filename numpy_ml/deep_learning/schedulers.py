import numpy as np

class CosineAnnealingLR():
    """
    Set the learning rate of each parameter group using a cosine annealing schedule
    without restart.

    Reference:
    https://arxiv.org/abs/1608.03983
    """
    def __init__(self, min_lr, t_max):
        """
        Parameters:
        ---------------
        min_lr: float
            Minimum learning rate
        t_max: int
            The period.
        """
        self.min_lr = min_lr
        self.t_max = t_max
        self.epoch = 0

    def __call__(self):
        return self.get_learning_rate()

    def get_max_lr(self, max_lr):
        self.max_lr = max_lr

    def get_learning_rate(self):
        curr_lr = self.min_lr+1/2*(self.max_lr-self.min_lr)*(1+np.cos(np.pi*self.epoch/self.t_max))
        self.epoch+=1
        return curr_lr

class CosineAnnealingWarmRestarts():
    """
    Set the learning rate of each parameter group using a cosine annealing schedule
    with restart.

    Reference:
    https://arxiv.org/abs/1608.03983
    """
    def __init__(self, min_lr, t_0, t_mult=1):
        self.min_lr = min_lr
        self.t_0=t_0
        self.t_mult=t_mult
        self.restart = False
        self.epoch = 0

    def __call__(self):
        return self.get_learning_rate()

    def get_max_lr(self, max_lr):
        self.max_lr = max_lr

    def get_learning_rate(self):
        if self.epoch >= self.t_0:
            self.epoch = self.epoch%self.t_0
            if self.epoch == 0:
                self.restart = True
        curr_lr = self.min_lr+1/2*(self.max_lr-self.min_lr)*(1+np.cos(np.pi*self.epoch/self.t_0))
        if self.restart:
            self.t_0 = self.t_0 * self.t_mult
            self.restart = False
        self.epoch += 1
        return curr_lr
