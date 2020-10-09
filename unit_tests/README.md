# Introduction

This is the part for unit tests mainly for deep learning. The main reason that I first write unit tests for deep learning is that deep learning is like Lego toys which consist of numerous subtle parts like various layers, various optimizers, various loss functions and so on so forth. It is hard to debug the entire net without first making sure that each part works correctly.

The ground truth of unit tests is based on [PyTorch](https://pytorch.org/) (the test of NCEloss function is based on Tensorflow) assuming the work of [PyTorch](https://pytorch.org/) is fully correct.

# How to run the tests

The test tool is [PyTest](https://docs.pytest.org/en/stable/).

Usage:
In the current folder, run:
```
pytest -s tests.py
```
<p align="center">
<img src="/images/unit_test.png">
</p>
<p align="center">
    The result of unit test for different parts of deep learning
</p>

The warning message is due to [the bug of Tensorflow](https://github.com/tensorflow/tensorflow/issues/31412)
