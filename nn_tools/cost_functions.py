import numpy as np


def quadratic(expected_vector, output_vector):
    error = 0.0
    for i, target in enumerate(expected_vector):
        error += 0.5 * (target - output_vector[i]) ** 2
    return error


# https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
def cross_entropy(expected_vector, output_vector):
    error = 0.0
    for i, target in enumerate(expected_vector):
        error += target * np.log(output_vector[i])
    return error * -1
