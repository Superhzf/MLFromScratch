import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits
import tensorflow as tf
# credits go to https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/tests/nn_torch_models.py
# I adjust the test and helper functions based on ddbourgin's work
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

def TFNCELoss(X, target_word, L):
    tf.compat.v1.disable_eager_execution()
    in_embed = tf.compat.v1.placeholder(tf.float32, shape=X.shape)
    in_bias = tf.compat.v1.placeholder(tf.float32, shape=L.b.flatten().shape)
    in_weights = tf.compat.v1.placeholder(tf.float32, shape=L.W.transpose().shape)
    in_target_word = tf.compat.v1.placeholder(tf.int64)
    in_neg_samples = tf.compat.v1.placeholder(tf.int32)
    in_target_prob = tf.compat.v1.placeholder(tf.float32)
    in_neg_samp_prob = tf.compat.v1.placeholder(tf.float32)

    feed = {
        in_embed: X,
        in_weights: L.W.transpose(),
        in_target_word: target_word,
        in_bias: L.b.flatten(),
        in_neg_samples: L.derived_variables["noise_samples"][0],
        in_target_prob: L.derived_variables["noise_samples"][1],
        in_neg_samp_prob: L.derived_variables["noise_samples"][2],
    }

    nce_unreduced = tf.nn.nce_loss(
        weights=in_weights,
        biases=in_bias,
        labels=in_target_word,
        inputs=in_embed,
        sampled_values=(in_neg_samples, in_target_prob, in_neg_samp_prob),
        num_sampled=L.num_negative_samples,
        num_classes=L.n_classes,
    )

    loss = tf.reduce_sum(nce_unreduced)
    dLdW = tf.gradients(loss, [in_weights])[0]
    dLdb = tf.gradients(loss, [in_bias])[0]
    dLdX = tf.gradients(loss, [in_embed])[0]

    sampled_logits, sampled_labels = _compute_sampled_logits(
        weights=in_weights,
        biases=in_bias,
        labels=in_target_word,
        inputs=in_embed,
        sampled_values=(in_neg_samples, in_target_prob, in_neg_samp_prob),
        num_sampled=L.num_negative_samples,
        num_classes=L.n_classes,
        num_true=1,
        subtract_log_q=True,
    )

    sampled_losses = sigmoid_cross_entropy_with_logits(
        labels=sampled_labels, logits=sampled_logits
    )

    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        (
            _final_loss,
            _nce_unreduced,
            _dLdW,
            _dLdb,
            _dLdX,
            _sampled_logits,
            _sampled_labels,
            _sampled_losses,
        ) = session.run(
            [
                loss,
                nce_unreduced,
                dLdW,
                dLdb,
                dLdX,
                sampled_logits,
                sampled_labels,
                sampled_losses,
            ],
            feed_dict=feed,
        )
    tf.compat.v1.reset_default_graph()

    return {
        "final_loss": _final_loss,
        "nce_unreduced": _nce_unreduced,
        "dLdW": _dLdW,
        "dLdb": _dLdb,
        "dLdX": _dLdX,
        "out_logits": _sampled_logits,
        "out_labels": _sampled_labels,
        "sampled_loss": _sampled_losses,
    }

class LSTM_many2many(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_many2many, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            bias=True,
                            num_layers=1,
                            bidirectional=False).double()
        # we will test a regression function, so no activation function is needed.
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=1,
                                bias=True).double()

    def forward(self, X):
        final_output = []
        # the shape of X is (seq_len, batch, input_size)
        lstm_output, (lstem_h, lstm_c) = self.lstm(X)
        # the shape of lstm_output is (seq_len, batch, hidden_size)
        for this_lstm_output in lstm_output:
            this_output = self.linear(this_lstm_output)
            final_output.append(this_output)
        return torch.stack(final_output)
