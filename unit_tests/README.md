# Introduction

This is the part for unit tests for my implementation including DL and non-DL algorithms.

The ground truth of unit tests for DL is based on [PyTorch](https://pytorch.org/) (the test of NCEloss function is based on [Tensorflow](https://www.tensorflow.org/)) assuming the work of [PyTorch](https://pytorch.org/) is fully correct. Regarding non-DL algorithms, I use [sklearn](https://scikit-learn.org/stable/) and [hmmlearn](https://github.com/hmmlearn/hmmlearn).

For sure, those established packages are not 100% correct (mature), for example the implementation of GaussianMixture model by sklearn learn
is not correct, https://github.com/scikit-learn/scikit-learn/issues/14419. Another example is decision trees by sklearn, https://github.com/scikit-learn/scikit-learn/pull/12364.

# How to run the tests

The test tool is [PyTest](https://docs.pytest.org/en/stable/).

Usage:
In the current folder, run:
```
pytest -s tests_dl.py
```
<p align="center">
<img src="/images/unit_test.png">
</p>
<p align="center">
    The result of unit test for different parts of deep learning
</p>

The warning message is due to [the bug of Tensorflow](https://github.com/tensorflow/tensorflow/issues/31412)
