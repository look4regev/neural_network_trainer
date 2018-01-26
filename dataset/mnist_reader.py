"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import os
import struct
import numpy as np
from matplotlib import pyplot

MAX_PIXEL_VALUE = 255.0


class DataSets(object):
    TRAINING = 'training'
    TESTING = 'testing'


def label_to_vector(label):
    vector = np.zeros(10)
    vector[label] = 1
    return vector


def _normalize_images(images):
    return [image/MAX_PIXEL_VALUE for image in images]


def read_mnist(dataset, path=".", read_max=100000):
    """Returns a dict of {labels, labels_vectors, images}"""

    if dataset == DataSets.TRAINING:
        images_file = os.path.join(path, 'train-images-idx3-ubyte.bin')
        labels_file = os.path.join(path, 'train-labels-idx1-ubyte.bin')
    elif dataset == DataSets.TESTING:
        images_file = os.path.join(path, 't10k-images-idx3-ubyte.bin')
        labels_file = os.path.join(path, 't10k-labels-idx1-ubyte.bin')
    else:
        raise ValueError("Dataset must be 'testing' or 'training'")

    with open(labels_file, 'rb') as label_file:
        struct.unpack(">II", label_file.read(8))
        labels = np.fromfile(label_file, dtype=np.int8)

    with open(images_file, 'rb') as image_file:
        _, _, rows, cols = struct.unpack(">IIII", image_file.read(16))
        images = np.fromfile(image_file, dtype=np.uint8).reshape(len(labels), rows*cols)
    images = _normalize_images(images)

    data = {
        'labels': labels[:read_max],
        'labels_vectors': [label_to_vector(label) for label in labels[:read_max]],
        'images': images[:read_max]
    }
    return data


def show(image):
    """Render a given numpy.uint8 2D array of pixel data."""
    fig = pyplot.figure()
    axis = fig.add_subplot(1, 1, 1)
    imgplot = axis.imshow(image)
    imgplot.set_interpolation('nearest')
    axis.xaxis.set_ticks_position('top')
    axis.yaxis.set_ticks_position('left')
    pyplot.show()
