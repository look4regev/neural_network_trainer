import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    """ The way we use this y is already sigmoided """
    return y * (1.0 - y)
