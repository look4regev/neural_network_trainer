#!/usr/bin/python
from dataset import mnist_reader
from nn_tools.neural_network import NeuralNetwork


MNIST_IMG_SIZE = 28
MNIST_CLASSES = 10
LAYERS_NEURONS_COUNT = {
    'input': MNIST_IMG_SIZE * MNIST_IMG_SIZE,
    'hidden': 3,
    'output': MNIST_CLASSES
}


def show_images(mnist):
    for image in mnist:
        mnist_reader.show(image)


def train_and_predict_mnist():
    mnist_training = mnist_reader.read('training', 'dataset/')
    mnist_testing = mnist_reader.read('testing', 'dataset/', with_labels=False)
    neural_network = NeuralNetwork(LAYERS_NEURONS_COUNT)
    neural_network.train(mnist_training)
    print 'Done training. Starting prediction.'
    predictions = neural_network.predict(mnist_testing)
    print "Predictions:"
    mnist_testing_labels = mnist_reader.read('testing', 'dataset/')
    for i, (label, _) in enumerate(mnist_testing_labels):
        print label, predictions[i]


if __name__ == "__main__":
    train_and_predict_mnist()
