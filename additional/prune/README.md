# Pruning

Pruning a CNN model. They are tested on the MNIST data set, which can be downloaded [here](https://www.kaggle.com/c/digit-recognizer). Note that this notebook requires PyTorch 1.4 or higher.

The pruning is based on thresholding, i.e. calculate a threshold value and set all weights lower than the threshold to zero.