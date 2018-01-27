# pylint: disable=no-self-use
import numpy as np
from cost_functions import ErrorCalculator

from nn_tools.neuron_tools import sigmoid, sigmoid_derivative


class NeuralNetwork(object):
    def __init__(self, input_layer_size, active_layers_sizes, l2_regularization=0):
        bias_increment = 1
        self.input = np.ones(input_layer_size + bias_increment)
        self.active = [np.ones(layer_size) for layer_size in active_layers_sizes]
        self.weights = []
        self._init_weights(active_layers_sizes)

        self.changes = []
        self._init_changes(active_layers_sizes)
        self.l2 = l2_regularization

    def get_output(self):
        return self.active[len(self.active)-1]

    def _init_weights(self, active_layers_sizes):
        weights_layer_rows_size = len(self.input)
        for layer_size in active_layers_sizes:
            self.weights.append(np.random.randn(weights_layer_rows_size, layer_size))
            weights_layer_rows_size = layer_size

    def _init_changes(self, active_layers_sizes):
        changes_layer_rows_size = len(self.input)
        for layer_size in active_layers_sizes:
            self.changes.append(np.zeros((changes_layer_rows_size, layer_size)))
            changes_layer_rows_size = layer_size

    def raise_on_wrong_inputs_count(self, input_layer_count):
        expected_size_without_bias = len(self.input) - 1
        if expected_size_without_bias != input_layer_count:
            raise ValueError('Wrong input size! Expected: %s, got %s' % (expected_size_without_bias,
                                                                         input_layer_count))

    def input_activation(self, input_layer):
        for i, element in enumerate(input_layer):
            self.input[i] = element

    def layer_activation(self, vector, matrix):
        result_vector = vector.dot(matrix)
        activated_vector = np.array([sigmoid(neuron_result) for neuron_result in result_vector])
        return activated_vector

    def feed_forward(self, input_layer):
        self.raise_on_wrong_inputs_count(input_layer.size)
        self.input_activation(input_layer)
        self.active[0] = self.layer_activation(self.input, self.weights[0])
        for i in range(len(self.active)-1):
            self.active[i+1] = self.layer_activation(self.active[i], self.weights[i+1])
        return self.get_output()

    def update_weights(self, layer1, layer2, delta, weights, changes, learning_rate):
        for j, layer1_elem in enumerate(layer1):
            for k in range(len(layer2)):
                change = delta[k] * layer1_elem
                regularization = self.l2 * weights[j][k]
                weights[j][k] -= learning_rate * (change + regularization) + changes[j][k]
                changes[j][k] = change

    def get_output_delta(self, target_vector):
        output_deltas = np.zeros(len(self.get_output()))
        for k in range(len(self.get_output())):
            error = -(target_vector[k] - self.get_output()[k])
            output_deltas[k] = sigmoid_derivative(self.get_output()[k]) * error
        return output_deltas

    def calc_layers_delta(self, layer1, layer2, weights):
        deltas = np.zeros(len(layer1))
        for j, layer1_elem in enumerate(layer1):
            error = layer2.dot(weights[j])
            deltas[j] = sigmoid_derivative(layer1_elem) * error
        return deltas

    def back_propagate(self, target_vector, learning_rate):
        """
        :param target_vector: y values
        :return: updates network weights and current error
        """
        if target_vector.size != len(self.get_output()):
            raise ValueError('Wrong number of targets!')
        # the delta tells you which direction to change the weights
        output_deltas = self.get_output_delta(target_vector)

        last_layer = self.get_output()
        last_deltas = output_deltas
        for i, hidden in enumerate(reversed(self.active[:-1])):
            weights = self.weights[len(self.weights) - i - 1]
            changes = self.changes[len(self.changes) - i - 1]
            self.update_weights(hidden, last_layer, last_deltas, weights, changes, learning_rate)

            last_deltas = self.calc_layers_delta(hidden, last_deltas, weights)
            last_layer = hidden

        self.update_weights(self.input, self.active[0], last_deltas,
                            self.weights[0], self.changes[0], learning_rate)

        return ErrorCalculator.get_error(target_vector, self.get_output())

    def train(self, labels, data, iterations, learning_rate=0.0002):
        error = 0.0
        for iteration in range(iterations):
            for data_index, label in enumerate(labels):
                self.feed_forward(data[data_index])
                error = self.back_propagate(label, learning_rate)
                if data_index % 50 == 0:
                    print 'Output error in iteration', iteration+1, 'and',\
                        data_index, 'trained items is %.5f' % error
            print 'Iteration', iteration+1, 'ended with error value of %.5f' % error

    def predict(self, input_vector):
        prediction_output_vector = self.feed_forward(input_vector)
        winning_class = np.argmax(prediction_output_vector)
        return winning_class
