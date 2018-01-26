#!/usr/bin/python

from dataset.mnist_reader import read_mnist, DataSets
from nn_tools.neural_network import NeuralNetwork


MNIST_IMG_SIZE = 28
MNIST_CLASSES_AS_OUTPUT_LAYER_SIZE = 10
INPUT_LAYER_NEURONS_COUNT = MNIST_IMG_SIZE * MNIST_IMG_SIZE
# Active means hidden layers + output layer
ACTIVE_LAYERS_NEURONS_COUNT = [20, 15, MNIST_CLASSES_AS_OUTPUT_LAYER_SIZE]
ITERATIONS = 5
MNIST_SUBSET_SIZE = 60000


def train_and_predict_mnist():
    neural_network = NeuralNetwork(INPUT_LAYER_NEURONS_COUNT, ACTIVE_LAYERS_NEURONS_COUNT)

    mnist_image_data = read_mnist(DataSets.TRAINING, 'dataset/', MNIST_SUBSET_SIZE)
    neural_network.train(mnist_image_data['labels_vectors'], mnist_image_data['images'], ITERATIONS)

    print 'Done training. Starting prediction on test set.'
    mnist_image_data = read_mnist(DataSets.TESTING, 'dataset/')
    matches = 0
    test_size = 0
    predictions_histogram = {key: 0 for key in range(MNIST_CLASSES_AS_OUTPUT_LAYER_SIZE)}
    for i, image_vector in enumerate(mnist_image_data['images']):
        prediction_label = neural_network.predict(image_vector)
        predictions_histogram[prediction_label] += 1
        if prediction_label == mnist_image_data['labels'][i]:
            matches += 1
        test_size += 1

    precision = float(matches) / test_size
    recall = float(test_size - matches) / test_size
    print "Done predictions on testset. Results:"
    print "Test set size =", test_size
    print "V) Matches =", matches
    print "X) Mistakes =", test_size - matches
    print "Total Precision = %.4f" % precision
    print "Total Recall = %.4f" % recall
    print 'Predictions histogram:', predictions_histogram


if __name__ == "__main__":
    train_and_predict_mnist()
