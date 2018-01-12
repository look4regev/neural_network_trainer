#!/usr/bin/python
from dataset import mnist_reader
from nn_tools.neural_network import NeuralNetwork


NEURONS_COUNT = 28  # Equals to mnist images width


def show_images(mnist):
    for image in mnist:
        mnist_reader.show(image)


def train_and_predict_mnist():
    mnist_training = mnist_reader.read('training', 'dataset/')
    mnist_testing = mnist_reader.read('testing', 'dataset/')
    neural_network = NeuralNetwork(NEURONS_COUNT)
    neural_network.train(mnist_training)
    print 'Done training. Starting prediction.'
    predictions = neural_network.predict(mnist_testing)
    print "Predictions:"
    print predictions


if __name__ == "__main__":
    train_and_predict_mnist()
