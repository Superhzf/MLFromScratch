import sys
sys.path.append('..')
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits
import tensorflow as tf
from numpy_ml.supervised_learning.decision_tree import Node
from numpy.testing import assert_almost_equal

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

class PyTorch_LSTM_many2many(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PyTorch_LSTM_many2many, self).__init__()
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

class PytorchDotProductAttention(nn.Module):
    def __init__(self,emb_dim, d_k=None, d_v=None):
        super(PytorchDotProductAttention, self).__init__()
        self.emb_dim = emb_dim
        if d_k is None:
            self.d_k = emb_dim
        else:
            self.d_k = d_k
        if d_v is None:
            self.d_v = emb_dim
        else:
            self.d_v = d_v

        self.softmax = nn.Softmax(dim=1)
        self.in_weight = nn.Linear(in_features=self.emb_dim, out_features=3*self.emb_dim, bias=False).double()
        self.out_weight = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=False).double()
        self.scale = 1/np.sqrt(self.emb_dim)

    def forward(self, Q, K, V):
        q,k,v = self.in_weight(Q).chunk(3, dim=-1)
        q = q*self.scale
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        k = k.transpose(1, 2)
        self.scores = torch.bmm(q, k)
        self.scores.retain_grad()
        target_len = self.scores.size(1)
        weights = []
        for this_target_len in range(target_len):
            this_weights = self.softmax(self.scores[:, this_target_len, :])
            weights.append(this_weights)
        weights = torch.stack(weights)
        self.weights = weights.transpose(0, 1)
        self.weights.retain_grad()
        outputs = torch.bmm(self.weights, v)
        outputs = outputs.transpose(0, 1)
        return self.out_weight(outputs), self.weights

# understanding the decision tree structure
# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.
# html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
def clone_tree(dtree):
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    feature = dtree.tree_.feature
    threshold = dtree.tree_.threshold
    values = dtree.tree_.value

    leaf_idx = 0
    def grow(node_id):
        nonlocal leaf_idx
        true_node = children_left[node_id]
        false_node = children_right[node_id]
        if true_node == false_node:
            leaf_idx+=1
            return Node(value=values[node_id].flatten()/np.sum(values[node_id]),
                        leaf_idx=leaf_idx-1)
        this_node = Node(feature_i=feature[node_id],threshold=threshold[node_id])
        this_node.true_branch = grow(true_node)
        this_node.false_branch = grow(false_node)
        return this_node

    node_id = 0
    root = Node(feature_i=feature[node_id],threshold=threshold[node_id])
    root.true_branch = grow(children_left[node_id])
    root.false_branch = grow(children_right[node_id])
    return root

def compare_trees(mine, gold):
    gold = clone_tree(gold)
    mine =  mine.root
    level = 0
    def test(mine, gold, level):
        if mine.value is None and gold.value is None:
            assert mine.feature_i == gold.feature_i,\
                   "Node feature at level {} are not equal".format(level)
            assert_almost_equal(mine.threshold,
                                gold.threshold,
                                4,
                                err_msg="Node threshold at level {} are not equal".format(level))
            test(mine.true_branch, gold.true_branch, level+1)
            test(mine.false_branch, gold.false_branch, level+1)
        elif mine.value is not None and gold.value is not None:
            assert_almost_equal(mine.value,
                                gold.value,
                                4,
                                err_msg="Node value at level {} are not equal".format(level))
            assert mine.leaf_idx == gold.leaf_idx,\
                   "leaf index at level {} are not equal".format(level)
            return
        else:
            level+=1
            raise ValueError("Nodes at level {} are not equal".format(level))

    test(mine, gold, level)
    return True
