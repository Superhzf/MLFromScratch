from __future__ import division
import numpy as np
from scipy.special import expit

class Loss(object):
    def loss(self,y_true,y_pred):
        pass

    def gradient(self,y_true,y_pred):
        pass

class SquareLoss(Loss):
    def __init__(self):
        pass

    def loss(self,y_true,y_pred):
        return 0.5*np.power((y_true-y_pred),2)

    def gradient(self,y_true,y_pred):
        return -(y_true-y_pred)

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


class NCELoss(Loss):
    def __init__(self, n_classes, noise_sampler, num_negative_samples, n_in=None,optimizer=None,subtract_log_label_prob=True):
        """
        A noise contrastive estimation (NCE) loss function. Noise contrastive
        estimation is a candidate sampling method often used to reduce the
        computational challenge of training a softmax layer on problems with a
        large number of output classes. It proceeds by training a logistic
        regression model to distriminate between samples from the true data
        distribution and samples from an artificial noise distribution.

        It can be shown that as the ration of negative samples to data samples
        goes into infinity, the gradient of the NCE loss converges to the original
        softmax gradient.

        Parameters:
        --------------------
        n_classes: int
            The total number of output classes in the model.
        noise_sampler: DiscreteSampler instance
            The negative sampler. Defines a distribution over all classes in
            the dataset.
        num_negative_samples: int
            The number of negative samples to draw for each target / batch of
            targets.
        optimizer: numpy_ml.deep_learning.optimizers object.
            The optimization strategy to use when performing gradient updates.
        subtract_log_label_prob: bool
            Whether to subtract the log of the probability of each label under
            the noise distribution from its respective logit. Set False for negative
            sampling, True for NCE.
        """
        self.n_in = n_in
        self.n_classes = n_classes
        self.noise_sampler = noise_sampler
        self.num_negative_samples = num_negative_samples
        self.act_fn = activation_functions["Sigmoid"]()
        self.optimizer = optimizer
        self.subtract_log_label_prob = subtract_log_label_prob
        self.trainable=True

    def initialize(self):
        self.X = []
        self.b = np.zeros(1, self.n_classes)
        limit = 1 / math.sqrt(self.n_in)
        self.W = np.random.uniform(-limit,limit,(self.n_in, self.n_classes))
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(b)

        self.derived_variables = {
            "y_pred": [],
            'target': [],
            'true_w': [],
            'true_b': [],
            'sampled_b': [],
            'sampled_w': [],
            'out_labels': [],
            'target_logits': [],
            'noise_samples': [],
            'noise_logits': []
        }

    def __call__(self, X, target, neg_samples=None, retain_derived=True):
        return self.loss(X, target, neg_samples, retain_derived)

    def loss(self, X, target, neg_samples=None, retain_derived=True):
        """
        Compute the NCE loss for a collection of inputs and associated targets.

        Parameters
        -------------------
        X: np.ndarray of shape (n_ex, n_c, n_in)
            n_ex observations,
        target: np.ndarray of shape (n_ex, )
            Target indices of the target classes for each example in the minibatch
            (e.g. the target word id for an example in a CBOW model)
        neg_samples: np.ndarray of shape (num_negative_samples, ) or None
            An optional array of negative samples to use during the loss calculation.
            These will be used instead of samples draw from self.noise_sampler
        retain_derived: bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, it suggests the layer will
            not be expected to do backprop with regard to this input.

        returns
        -------------------
        loss: float
            The NCE loss summed over the minibatch and samples.
        y_pred: np.ndarray of shape (n_ex, n_c)
            The network predictions for the conditional probability of each target
            given each context: entry (i, j) gives the predicted probability of
            target i under context vector j
        """
        if self.n_in is None:
            self.n_in = X.shape[-1]

        loss, Z_target, Z_neg, y_pred, y_true, noise_samples = self._loss(
            X, target, neg_samples
        )

        if retain_derived:
            self.X.append(X)

            self.derived_variables['y_pred'].append(y_pred)
            self.derived_variables['target'].append(target)
            self.derived_variables['out_labels'].append(y_true)
            self.derived_variables['target_logits'].append(Z_target)
            self.derived_variables['noise_samples'].append(noise_samples)
            self.derived_variables['noise_logits'].append(Z_neg)

        return loss, np.squeeze(y_pred[..., :1], -1)

    def _loss(self, X, target, neg_samples):
        fstr = "X must have shape (n_ex, n_c, n_in), but got {} dims instead"
        assert X.ndim == 3, fstr.format(X.ndim)

        W = self.W
        b = self.b

        if neg_samples is None:
            neg_samples = self.noise_sampler(self.num_negative_samples)
        assert len(neg_samples) == self.num_negative_samples

        # Get the probability of the negative sample class and the target
        # class under the noise distribution
        p_neg_samples = self.noise_sampler.probs[neg_samples]
        p_target = np.atleast_2d(self.noise_sampler.probs[target])

        # Save the noise samples for debugging
        noise_samples = (neg_samples, p_target, p_neg_samples)

        # Compute the logit for the negative samples and target
        Z_target = X @ W[target].T + b[0, target]
        Z_neg = X @ W[neg_samples].T + b[0, neg_samples]

        # subtract the log probability of each label under the noise dist
        if self.subtract_log_label_prob:
            n = Z_target.shape[0]
            m = Z_neg.shape[0]
            Z_target[range(n), ...] -= np.log(p_target)
            Z_neg[range(m), ...] -= np.log(p_neg_samples)

        # only retain the probability of the target under its associated
        # minibatch example
        aa, _, cc = Z_target.shape
        Z_target = Z_target[range(aa), :, range(cc)][..., None]

        pred_p_target = self.act_fn(Z_target)
        pred_p_neg = self.act_fn(Z_neg)

        # If we are in evaluation mode, ignore the negative samples - just
        # return the binary cross entropy on the targets
        y_pred = pred_p_target
        if self.trainable:
            # (n_ex, n_c, 1 + n_samples) (target is the first column)
            y_pred = np.concatenate((y_pred, pred_p_neg), axis=-1)

        n_targets = 1
        y_true = np.zeros_like(y_pred)
        y_true[..., :n_targets] = 1

        # binary cross entropy
        eps = np.finfo(float).eps
        np.clip(y_pred, eps, 1-eps, y_pred)
        loss = -np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return loss, Z_target, Z_neg, y_pred, y_true, noise_samples

    def gradient(self, retain_grads=True, update_params=True):
        """
        Compute the gradient of the NCE loss with regard to the inputs, weights,
        and bias

        Parameters:
        --------------------
        retain_grads: bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update.
        update_params: bool
            Whether to perform a single step of gradient descent on the layer
            weights and bias using calculated gradients. If retain_grads is False,
            this option is ignored and the parameter gradients are not updated.

        Returns:
        -------------------
        dLdX: np.ndarray of (n_ex, n_in) or list of array
            The gradient of the loss w.r.t. the layer inputs
        """
        dX = []
        for input_idx, x in enumerate(self.X):
            dx, dw, db = self._grad(x, input_idx)
            dX.append(dx)

            if retain_grads:
                self.dW += dw
                self.db += db

        dX = dX[0] if len(self.X) == 1 else dX

        if retain_grads and update_params:
            pass
            # do update

        return dX

    def _grad(self, X, input_idx):
        W = self.W
        b = self.b

        y_pred = self.derived_variables["y_pred"][input_idx]
        target = self.derived_variables["target"][input_idx]
        y_true = self.derived_variables["out_labels"][input_idx]
        Z_neg = self.derived_variables["noise_logits"][input_idx]
        Z_target = self.derived_variables["target_logits"][input_idx]
        neg_samples = self.derived_variables["noise_samples"][input_idx][0]

        # the number of target classes per minibatch example
        n_targets = 1

        # calculate the grad of the binary cross entropy w.r.t. the network
        # predictions
        preds = y_pred.flattern()
        classes = y_true.flatten()

        dLdp_real = ((1 - classes) / (1 - preds)) - (classes / preds)
        dLdp_real = dLdp_real.reshape(*y_pred.shape)

        # partition the gradients into target and negative sample portions
        dLdy_pred_target = dLdp_real[..., :n_targets]
        dLdy_pred_neg = dLdp_real[..., n_targets:]

        # compute gradients of the loss wrt the data and noise logits
        dLdZ_target = dLdy_pred_target * self.act_fn.gradient(Z_target)
        dLdZ_neg = dLdy_pred_neg * self.act_fn.grad(Z_neg)

        # compute param gradients on target + negative samples
        dB_neg = dLdZ_neg.sum(axis=(0, 1))
        dB_target = dLdZ_target.sum(axis=(1, 2))


        dW_neg = (dLdZ_neg.transpose(0, 2, 1) @ X).sum(axis=0)
        dW_target = (dLdZ_target.transpose(0, 2, 1) @ X).sum(axis=0)

        dX_target = np.vstack(
            [dLdZ_target[[ix]] @ W[[t]] for ix, t in enumerate(target)]
        )
        dX_neg = dLdZ_neg @ W[neg_samples]

        hits = list(set(target)).intersection(set(neg_samples))
        hit_ixs = [np.where(target == h)[0] for h in hits]

        # adjust param gradients if there is an accidental hit
        if len(hits)!=0:
            hit_ixs = np.concatenate(hit_ixs)
            target = np.delete(target, hit_ixs)
            dB_target = np.delete(dB_target, hit_ixs)
            dW_target = np.delete(dW_target, hit_ixs, 0)

        dX = dX_target + dX_neg
        # use np.add.at to ensure that repeated indices in the target (or
        # possibly in neg_samples if sampling is done with replacement) are
        # properly accounted for
        dB = np.zeros_like(b).flatten()
        np.add.at(dB, target, dB_target)
        np.add.at(dB, neg_samples, dB_neg)
        dB = dB.reshape(*b.shape)

        dW = np.zeros_like(W)
        np.add.at(dW, target, dW_target)
        np.add.at(dW, neg_samples, dW_neg)

        return dX, dW, dB
