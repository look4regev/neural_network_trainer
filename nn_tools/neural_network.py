# pylint: disable=no-self-use
import numpy as np

from nn_tools.neuron_tools import sigmoid, sigmoid_derivative


ITERATIONS = 3000


class NeuralNetwork(object):
    def __init__(self, layers_count):
        bias_increment = 1
        self.input_count = layers_count['input'] + bias_increment
        self.output_count = layers_count['output']
        self.layers_count = layers_count

        self.layers = {'input': np.ones(self.input_count),
                       'output': np.ones(self.output_count),
                       'hidden': np.ones(self.layers_count['hidden'])}

        self.weights_input = np.random.randn(self.input_count, self.layers_count['hidden'])
        self.weights_output = np.random.randn(self.layers_count['hidden'], self.output_count)
        # create arrays of 0 for changes
        self.change_inputs = np.zeros((self.input_count, self.layers_count['hidden']))
        self.change_outputs = np.zeros((self.layers_count['hidden'], self.output_count))

    def raise_on_wrong_inputs_count(self, inputs_count):
        if inputs_count != self.input_count - 1:
            raise ValueError('Wrong number of inputs!')

    def input_activation(self, input_values):
        for i in range(self.input_count - 1):  # -1 is to avoid the bias
            self.layers['input'][i] = input_values[i]

    def layer_activation(self, vector, matrix):
        result_vector = vector.dot(matrix)
        return np.array([sigmoid(neuron_result) for neuron_result in result_vector])

    def feed_forward(self, input_layer):
        self.raise_on_wrong_inputs_count(input_layer.size)
        self.input_activation(input_layer)
        self.layers['hidden'] = self.layer_activation(self.layers['input'], self.weights_input)
        self.layers['output'] = self.layer_activation(self.layers['hidden'], self.weights_output)
        return self.layers['output'][:]

    def back_propagate(self, targets, learning_rate):
        """
        :param targets: y values
        :return: updated weights and current error
        """
        if targets.size != self.output_count:
            raise ValueError('Wrong number of targets you silly goose!')
        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = np.zeros(self.output_count)
        for k in range(self.output_count):
            error = -(targets[k] - self.layers['output'][k])
            output_deltas[k] = sigmoid_derivative(self.layers['output'][k]) * error
        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = np.zeros(self.layers_count['hidden'])
        for j in range(self.layers_count['hidden']):
            error = 0.0
            for k in range(self.output_count):
                error += output_deltas[k] * self.weights_output[j][k]
            hidden_deltas[j] = sigmoid_derivative(self.layers['hidden'][j]) * error
        # update the weights connecting hidden to output
        for j in range(self.layers_count['hidden']):
            for k in range(self.layers_count['hidden']):
                change = output_deltas[k] * self.layers['hidden'][j]
                self.weights_output[j][k] -= learning_rate * change + self.change_outputs[j][k]
                self.change_outputs[j][k] = change
        # update the weights connecting input to hidden
        for i in range(self.input_count):
            for j in range(self.layers_count['hidden']):
                change = hidden_deltas[j] * self.layers['input'][i]
                self.weights_input[i][j] -= learning_rate * change + self.change_inputs[i][j]
                self.change_inputs[i][j] = change
        # calculate error
        error = 0.0
        for i, target in enumerate(targets):
            error += 0.5 * (target - self.layers['output'][i]) ** 2
        return error

    def label_to_output_array(self, label):
        result = np.zeros(10)
        result[label] = 1
        return result

    def train(self, patterns, iterations=ITERATIONS, learning_rate=0.0002):
        for i in range(iterations):
            error = 0.0
            for j, (label, input_layer) in enumerate(patterns):
                self.feed_forward(input_layer)
                error = self.back_propagate(self.label_to_output_array(label), learning_rate)
                if j % 50 == 0:
                    print 'error %-.5f' % error
            if i % 500 == 0:
                print 'error %-.5f' % error

    def predict(self, items):
        """
        :return: list of predictions after training algorithm
        """
        predictions = []
        for input_layer in items:
            predictions.append(self.feed_forward(input_layer))
        return predictions
