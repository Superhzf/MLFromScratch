# Machine learning from scratch
Numpy-ml from scratch. This repo aims to help myself/people understand the math behind
machine learning algorithms and I will try to make the computation as
efficient as possible

# Implementations

## Supervised Learning

- [Decision Tree](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/decision_tree.py)
- [Grdient Boosting Tree](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/gradient_boosting.py)
- [Logistic Regression](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/logistic_regression.py)
- [Linear Regression](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py)
- [Elastic Net](https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py)
- [Ridge Regression](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/regression.py)
- [Lasso Regression](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/regression.py)
- [Support Vector Machine](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/support_vector_machine.py)
- [Xgboost](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/supervised_learning/xgboost.py)

## Deep Learning

- [Neural Network](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/neural_network.py)
- [Layers](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/layers.py)

  * Activation Layer
  * Batch Normalization Layer
  * Dropout Layer
  * Fully Connected Layer
  * Embedding Layer
  * RNN Layer: many-to-one
  * LSTM ayer: many-to-one
  * Bidirectional LSTM
- [Loss Functions](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/loss_functions.py)

  * Cross Entropy
  * Loss for VAE
  * BinomialDeviance
  * Noise Contrastive Estimation

- [Optimizer](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/optimizers.py)

  * SGD with momentum
  * RMSprop
  * Adagrad
  * Adadelta
  * Adam

- [Schedulers](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/schedulers.py)
  * CosineAnnealingLR
  * CosineAnnealingWarmRestarts

- Models
  * [word2vec](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/deep_learning/models/word2vec.py)
  * LSTM many to many

## Unsupervised Learning

- [Generative Adversarial Network](https://github.com/Superhzf/MLFromScratch/blob/master/numpy_ml/unsupervised_learning/generative_adversarial_network.py)

# Examples

### SVM
<p align="center">
<img src="/images/svm.png">
</p>

### Polynomial Lasso Regression
<p align="center">
<img src="/images/poly_lasso_regress.png">
</p>


### Decision Tree for Classification
<p align="center">
<img src="/images/decision_tree_classification.png">
</p>

### Decision Tree for Regression
<p align="center">
<img src="/images/decision_tree_regression.png">
</p>

### Xgboost
<p align="center">
<img src="/images/xgb.png">
</p>

### deep learning

<p align="center">
<img src="/images/unit_test.png">
</p>
<p align="center">
    The result of unit test for different parts of deep learning
</p>

The warning message is due to [the bug of Tensorflow](https://github.com/tensorflow/tensorflow/issues/31412)

[unit test page](https://github.com/Superhzf/MLFromScratch/tree/master/unit_tests)
