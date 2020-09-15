import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
# credits go to https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/tests/nn_torch_models.py
def random_one_hot_matrix(n_examples, n_classes):
    """Create a random one-hot matrix of shape (`n_examples`, `n_classes`)"""
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X

def random_stochastic_matrix(n_examples, n_classes):
    """Create a random stochastic matrix of shape (`n_examples`, `n_classes`)"""
    X = np.random.rand(n_examples, n_classes)
    X /= X.sum(axis=1, keepdims=True)
    return X

def random_tensor(shape, standardize=False):
    """
    Create a random real-valued tensor of shape `shape`. If `standardize` is
    True, ensure each column has mean 0 and std 1.
    """
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def torchify(var, requires_grad=True):
    return torch.autograd.Variable(torch.DoubleTensor(var), requires_grad=requires_grad)

def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg

class TorchVAELoss(nn.Module):
    def __init__(self):
        super(TorchVAELoss, self).__init__()

    def extract_grads(self, X, X_recon, t_mean, t_log_var):
        eps = np.finfo(float).eps
        X = torchify(X, requires_grad=False)
        X_recon = torchify(np.clip(X_recon, eps, 1 - eps))
        t_mean = torchify(t_mean)
        t_log_var = torchify(t_log_var)

        BCE = torch.sum(F.binary_cross_entropy(X_recon, X, reduction="none"), dim=1)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + t_log_var - t_mean.pow(2) - t_log_var.exp(), dim=1)

        loss = torch.mean(BCE + KLD)
        loss.backward()

        grads = {
            "loss": loss.detach().numpy(),
            "dX_recon": X_recon.grad.numpy(),
            "dt_mean": t_mean.grad.numpy(),
            "dt_log_var": t_log_var.grad.numpy(),
        }
        return grads
