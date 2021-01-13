from __future__ import division
import numpy as np
from scipy.special import expit
from .activation_functions import Sigmoid
import math
import copy

class Loss(object):
    def loss(self,y_true,y_pred):
        pass

    def gradient(self,y_true,y_pred):
        pass

class SquaredLoss(Loss):
    def __init__(self):
        pass

    def loss(self,y_true,y_pred):
        return np.power((y_true-y_pred),2)

    def gradient(self,y_true,y_pred):
        return -2*(y_true-y_pred)

# Why Cross Entropy loss instead of MSE for binary classification problems?
# Answer: It is OK to use MSE but CE loss is better, the reason is that CE
# penalizes much to incorrect predictions, image the true value is 0 but your
# prediction is 1.

# Why CE loss generally?
# Answer: Because minimizing cross entropy loss is equal to maximizing log likelihood
# Proof: if y_true = 1, suppose P(y|x) = y_pred, if y_true = 0, P(y|x) = 1-y_pred
# If we combine them, we have P(y|x) = (y_pred^y)*(1-y_pred)^(1-y)
# logP(y|x) = log(y_pred^y) + log(1-y_pred)^(1-y) = ylogy+(1-y)log(1-y)
class BinaryCrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self,y,p):
        assert y.shape[1] == 2, 'BinaryCrossEntropy can only be used for binary classification problems'
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y*np.log(p)+(1-y)*np.log(1-p))

    def gradient(self,y,p):
        # Avoid zero numerator
        p = np.clip(p,1e-15,1-1e-15)
        return - (y/p)+(1-y)/(1-p)

class CrossEntropy(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y * np.log(p + eps),axis=1)
        return cross_entropy

    def gradient(self, y, p):
        # eps = np.finfo(float).eps
        # return -np.sum(y/(p+eps),axis=1)
        # The gradient here is the gradient of the cross entropy loss w.r.t.
        # the input of softmax because the calculation is much easier
        return p - y


class BinomialDeviance(Loss):
    def __init__(self):
        pass

    def loss(self, y, p):
        return -2 * np.mean((y * p) - np.logaddexp(0, p))

    def negative_gradient(self, y, p):
        return y - expit(p.ravel())

    def update_terminal_region(self, X, y, residual, tree_model):
        idx_list = np.array([tree_model.apply(sample) for sample in X])
        stack = [tree_model.root]
        node_list = []
        while len(stack) > 0:
            curr = stack.pop()
            if curr.value is not None:
                node_list.append(curr)
            if curr.true_branch is not None:
                stack.append(curr.true_branch)
            if curr.false_branch is not None:
                stack.append(curr.false_branch)

        for this_node in node_list:
            this_y = y[idx_list == this_node.leaf_idx]
            this_residual = residual[idx_list == this_node.leaf_idx]
            assert len(this_y) == len(this_residual) and len(this_y) > 0
            # The fomula comes from Newton's method and Taloyr extension
            numerator = np.sum(this_residual)
            denominator = np.sum((this_y - this_residual) * (1 - this_y + this_residual))
            if abs(denominator) < 1e-150:
                this_node.value = 0.0
            else:
                this_node.value = numerator / denominator


class VAELoss(Loss):
    def __init__(self):
        """
        The variational lower bound for a variational autoencoder

        VAELoss = cross_entropy(y, y_hat) + KL(q||p), p is a unit Gaussian distribution
        (0, 1) and q = q(Z|X) follows a Gaussian distribution N(mean(X), std(X))

        KL(q||p) is a regularization term enforcing the latent distributions
        to be close to the standard normal distribution. In that case, the VAE
        model would be able to generate new cases.
        """
        pass

    def loss(self, y, y_pred, t_mean, t_log_var):
        """
        Variational lower bound for VAE.

        parameters:
        ----------------
        y: np.ndarray of shape (n_ex, N).
            The n_ex number of original images with N pixels.
        y_pred: np.ndarray of shape (n_ex, N)
            The n_ex number of VAE reconstruction images with N pixels.
        t_mean: np.ndarray of shape (n_ex, T)
            Mean of the encoder q(Z|X)
        t_log_var: np.ndarray of shape (n_ex, T)
            Log of the variance vector of the encoder q(Z|X)

        Return:
        -----------------
        loss: float
            The variational lower bound across the batch
        """
        # prevent nan on log(0)
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # reconstruction loss
        rec_loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred), axis=1)

        # KL divergence between q and p
        kl_loss = -0.5 * np.sum(1 + t_log_var - t_mean ** 2 - np.exp(t_log_var), axis=1)

        loss = kl_loss + rec_loss
        return loss

    def gradient(self, y, y_pred, t_mean, t_log_var):
        """
        Compute the graident of the variational lower bound with respect to the
        network parameters.

        parameters:
        ----------------
        y: np.ndarray of shape (n_ex, N)
            The n_ex number of original images with N pixels.
        y_pred: np.ndarray of shape (n_ex, N)
            The n_ex number of VAE reconstruction images with N pixels.
        t_mean: np.ndarray of shape (n_ex, T)
            Mean of the encoder q(Z|X)
        t_log_var: np.ndarray of shape (n_ex, T)
            Log of the variance vector of the encoder q(Z|X)

        Returns:
        ---------------
        dY_pred: np.ndarray of (n_ex, N)
            The gradient of the lower bound with respect to y_pred
        dMean: np.ndarray of (n_ex, T)
            The gradient of the lower bound wih respect to t_mean
        dLogVar: np.ndarray of (n_ex, T)
            The gradient of the lower bound wih respect to t_log_var
        """
        N = y.shape[0]
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)

        dY_pred = (-y / (N * y_pred) - (y - 1) / (N - N * y_pred))
        dMean = t_mean / N
        dLogVar = (np.exp(t_log_var) - 1) / (2 * N)
        return dY_pred, dMean, dLogVar


class NCELoss(Loss):
    def __init__(self,
                 n_classes,
                 n_in,
                 noise_sampler,
                 num_negative_samples,
                 subtract_log_label_prob=True,
                 trainable=True):
        """
        Noise contrastive estimation (NCE) function. NCE is a candidate sampling
        method often used to reduce the computational challenge of training a
        softmax layer on problems with a large number of output classes. It proceeds
        by training a logistic regression model to discriminate between samples
        from true data distribution and samples from an artificial noise distribution.

        Parameter:
        ----------------
        n_classes: int
            The total number of output classes in the model.
        n_in: int
            The number of features of the input X. It is the embedding_dim for
            word2vec model.
        noise_sampler:
            The negative sampler. Defines a distribution over all classes in the
            dataset.
        num_negative_samples: int
            The number of negative samples to draw for each target.
        subtract_log_label_prob: bool
            Whether to subtract the log of the probability of each label under the
            noise distribution from its respective logit. Set to false for negative
            sampling, true for NCE.
        trainable: bool
            Whether to update the weights and bias items in the backward
            propogation.
        """
        self.n_in=n_in
        self.trainable=trainable
        self.n_classes=n_classes
        self.noise_sampler=noise_sampler
        self.num_negative_samples=num_negative_samples
        self.subtract_log_label_prob = subtract_log_label_prob

    def initialize(self, optimizer):
        self.optimizer = optimizer
        self.act_fn=Sigmoid()
        self.X = []

        limit = 1 / math.sqrt(self.n_in)
        self.W = np.random.uniform(-limit,limit,(self.n_in,self.n_classes))
        self.b = self.b = np.zeros((1,self.n_classes))

        self.W_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.derived_variables = {
            "y_pred":[],
            "target":[],
            "out_labels":[],
            "target_logits":[],
            "noise_samples":[],
            "noise_logits":[]
        }

    def loss(self, X, target, neg_samples=None, train=True):
        """
        Compute the NCE loss for a collection of inputs and associated targets.

        Parameters:
        ----------------------
        X: numpy.array of shape (n_ex, n_c, n_in)
            The layer input. A minibatch of n_ex observations, where each observation
            has n_c features (characters) and each character has n_in output
            features.
        target: numpy.array of shape (n_ex,)
            Integer indices (ID) of target classes for each example in the minibatch.
        neg_sampels: numpy.array of shape (num_negative_samples,)
            An optional array of negative samples to use during the loss calculation.
            These will be used instead of samples draw from self.noise_sampler
        train: bool
            Whether it is the training process or evaluation process. If true,
            then the output will include the prediction for both the positive
            targets and negative targets. Otherwise, only the positive target
            prediction will be included in the output.

        Returns:
        -----------------
        loss: float
            The NCE loss summed over the minibatch and samples.
        y_pred: numpy.array of shape (n_ex, n_c):
            The network predictions
        """
        loss, Z_target, Z_neg, y_pred, y_true, noise_samples=self._loss(X, target, neg_samples, train)

        self.X.append(X)
        self.derived_variables['y_pred'].append(y_pred)
        self.derived_variables['target'].append(target)
        self.derived_variables['out_labels'].append(y_true)
        self.derived_variables['target_logits'].append(Z_target)
        self.derived_variables['noise_samples'].append(noise_samples)
        self.derived_variables['noise_logits'].append(Z_neg)

        return loss, np.squeeze(y_pred[..., :1], -1)


    def _loss(self, X, target, neg_samples, train=True):
        """Actual calculation of NCE loss"""
        assert len(X.shape)==3

        if neg_samples is None:
            neg_samples = self.noise_sampler(self.num_negative_samples)
        assert len(neg_samples) == self.num_negative_samples

        # neg_samples are the negative targets, negative sampels share the same
        # inputs as positive samples.
        p_neg_samples = self.noise_sampler.probs[neg_samples]
        # target are the positive targets
        p_target = np.atleast_2d(self.noise_sampler.probs[target])

        # pair up positive samples and negative samples
        noise_samples = (neg_samples, p_target, p_neg_samples)

        # compute the logit for negative samples and target
        Z_target = X @ self.W[:,target] + self.b[:,target]
        Z_neg = X @ self.W[:,neg_samples] + self.b[:,neg_samples]

        # subtract the log probability of each label under the noise dist
        if self.subtract_log_label_prob:
            n, m = Z_target.shape[0], Z_neg.shape[0]
            Z_target[range(n), ...] -= np.log(p_target)
            Z_neg[range(m), ...] -= np.log(p_neg_samples)

        aa, _, cc = Z_target.shape
        Z_target = Z_target[range(aa), :, range(cc)][..., None]

        pred_p_target = self.act_fn(Z_target)
        pred_p_neg = self.act_fn(Z_neg)

        y_pred = pred_p_target
        if train:
            # (n_ex, n_c, 1 + n_samples) (target is first column)
            y_pred = np.concatenate((y_pred, pred_p_neg), axis=-1)

        # we have only one positive samples
        n_targets = 1
        y_true = np.zeros_like(y_pred)
        y_true[..., :n_targets] = 1

        # binary cross entropy
        eps = np.finfo(float).eps
        np.clip(y_pred, eps, 1 - eps, y_pred)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss, Z_target, Z_neg, y_pred, y_true, noise_samples

    def gradient(self):
        dX = []
        for idx, x in enumerate(self.X):
            dx, dW, db = self._gradient(x, idx)
            dX.append(dx)

            self.dW += dW
            self.db += db

        dX = dX[0] if len(self.X) == 1 else dX
        if self.trainable:
            self.W = self.W_opt.update(self.W, self.dW)
            self.b = self.b_opt.update(self.b, self.db)

            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)

        return np.stack(dX)

    def _gradient(self, X, idx):

        y_pred = self.derived_variables['y_pred'][idx]
        target = self.derived_variables['target'][idx]
        y_true = self.derived_variables['out_labels'][idx]
        Z_neg = self.derived_variables['noise_logits'][idx]
        Z_target = self.derived_variables['target_logits'][idx]
        neg_samples = self.derived_variables['noise_samples'][idx][0]

        # the number of target classes per minibatch example
        n_targets = 1

        preds, classes = y_pred.flatten(), y_true.flatten()

        dLdp = ((1 - y_true) / (1 - y_pred)) - (y_true / y_pred)
        # is this necessary???
        # dLdp = dLdp.reshape(*y_pred.shape)

        # partition the gradients into target and negative sample portions
        dLdy_pred_target = dLdp[..., :n_targets]
        dLdy_pred_neg = dLdp[..., n_targets:]

        # compute gradients of the loss wrt the data and noise logits
        dLdZ_target = dLdy_pred_target * self.act_fn.gradient(Z_target)
        dLdZ_neg = dLdy_pred_neg * self.act_fn.gradient(Z_neg)

        # compute weight gradients
        db_neg = dLdZ_neg.sum(axis=(0, 1))
        db_target = dLdZ_target.sum(axis=(1, 2))

        dW_neg = (dLdZ_neg.transpose(0, 2, 1) @ X).sum(axis=0)
        dW_target = (dLdZ_target.transpose(0, 2, 1) @ X).sum(axis=1)

        # compute gradients w.r.t. the input X
        dX_target = np.vstack([dLdZ_target[[ix]] @ self.W[:,[t]].transpose() for ix, t in enumerate(target)])
        dX_neg = dLdZ_neg @ self.W[:,neg_samples].transpose()

        hits = list(set(target).intersection(set(neg_samples)))
        hit_ixs = [np.where(target == h)[0] for h in hits]

        # adjust param gradients if there's an accidental hit
        if len(hits) != 0:
            hit_ixs = np.concatenate(hit_ixs)
            target = np.delete(target, hit_ixs)
            db_target = np.delete(db_target, hit_ixs)
            dW_target = np.delete(dW_target, hit_ixs, 0)

        dX = dX_target + dX_neg

        db = np.zeros_like(self.b).flatten()
        np.add.at(db, target, db_target)
        np.add.at(db, neg_samples, db_neg)
        db = db.reshape(*self.b.shape)

        dW = np.zeros_like(self.W).transpose()

        np.add.at(dW, target, dW_target)
        np.add.at(dW, neg_samples, dW_neg)
        dW = dW.transpose()

        return dX, dW, db
