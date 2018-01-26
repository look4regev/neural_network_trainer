import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(y):
    """ The way we use this y is already sigmoided """
    return y * (1.0 - y)


def softmax(weights):
    diff_exponent = np.exp(weights - np.amax(weights))
    dist = diff_exponent / np.sum(diff_exponent)
    return dist


def tanh(x):
    return np.tanh(x)


def tanh_derivative(y):
    return 1 - y*y
