"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import os
import struct
import numpy as np
from matplotlib import pyplot


def read(dataset="training", path=".", with_labels=True):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte.bin')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte.bin')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte.bin')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte.bin')
    else:
        raise ValueError("Dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)

    def get_img(idx):
        if with_labels:
            return lbl[idx], img[idx]
        return img[idx]

    for i in xrange(len(lbl)):
        yield get_img(i)


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    axis = fig.add_subplot(1, 1, 1)
    imgplot = axis.imshow(image)
    imgplot.set_interpolation('nearest')
    axis.xaxis.set_ticks_position('top')
    axis.yaxis.set_ticks_position('left')
    pyplot.show()
