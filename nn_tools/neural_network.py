import numpy as np

from nn_tools.neuron_tools import sigmoid, sigmoid_derivative


class NetworkValues(object):
    def __init__(self, count):
        self.input_neuron_count = count + 1  # Bias node addition
        self.hidden_neuron_count = count
        self.output_neuron_count = count
        # set up array of 1s for activations
        self.inputs = [1.0] * self.input_neuron_count
        self.hidden = [1.0] * self.hidden_neuron_count
        self.outputs = [1.0] * self.output_neuron_count

    def raise_on_wrong_inputs_count(self, inputs_count):
        if inputs_count != self.input_neuron_count - 1:
            raise ValueError('Wrong number of inputs!')

    def input_activation(self, inputs):
        for i in range(self.input_neuron_count - 1):  # -1 is to avoid the bias
            self.inputs[i] = inputs[i]

    def hidden_activation(self, rand_weight_inputs):
        for j in range(self.hidden_neuron_count):
            sum_hidden = 0.0
            for i in range(self.input_neuron_count):
                sum_hidden += self.inputs[i] * rand_weight_inputs[i][j]
            self.hidden[j] = sigmoid(sum_hidden)

    def output_activation(self, rand_weight_outputs):
        for k in range(self.output_neuron_count):
            sum_output = 0.0
            for j in range(self.hidden_neuron_count):
                sum_output += self.hidden[j] * rand_weight_outputs[j][k]
            self.outputs[k] = sigmoid(sum_output)


class NeuralNetwork(object):
    def __init__(self, neuron_count):
        self.network_values = NetworkValues(neuron_count)
        # create randomized weights
        self.rand_weight_inputs = np.random.randn(self.network_values.input_neuron_count,
                                                  neuron_count)
        self.rand_weight_outputs = np.random.randn(neuron_count, neuron_count)
        # create arrays of 0 for changes
        self.change_inputs = np.zeros((self.network_values.input_neuron_count, neuron_count))
        self.change_outputs = np.zeros((neuron_count, neuron_count))

    def feed_forward(self, inputs):
        self.network_values.raise_on_wrong_inputs_count(len(inputs))
        self.network_values.input_activation(inputs)
        self.network_values.hidden_activation(self.rand_weight_inputs)
        self.network_values.output_activation(self.rand_weight_outputs)
        return self.network_values.outputs[:]

    def back_propagate(self, targets, learning_rate):
        """
        :param targets: y values
        :return: updated weights and current error
        """
        if len(targets) != self.network_values.output_neuron_count:
            raise ValueError('Wrong number of targets you silly goose!')
        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.network_values.output_neuron_count
        for k in range(self.network_values.output_neuron_count):
            error = -(targets[k] - self.network_values.outputs[k])
            output_deltas[k] = sigmoid_derivative(self.network_values.outputs[k]) * error
        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.network_values.hidden_neuron_count
        for j in range(self.network_values.hidden_neuron_count):
            error = 0.0
            for k in range(self.network_values.output_neuron_count):
                error += output_deltas[k] * self.rand_weight_outputs[j][k]
            hidden_deltas[j] = sigmoid_derivative(self.network_values.hidden[j]) * error
        # update the weights connecting hidden to output
        for j in range(self.network_values.hidden_neuron_count):
            for k in range(self.network_values.output_neuron_count):
                change = output_deltas[k] * self.network_values.hidden[j]
                self.rand_weight_outputs[j][k] -= learning_rate * change + self.change_outputs[j][k]
                self.change_outputs[j][k] = change
        # update the weights connecting input to hidden
        for i in range(self.network_values.input_neuron_count):
            for j in range(self.network_values.hidden_neuron_count):
                change = hidden_deltas[j] * self.network_values.inputs[i]
                self.rand_weight_inputs[i][j] -= learning_rate * change + self.change_inputs[i][j]
                self.change_inputs[i][j] = change
        # calculate error
        error = 0.0
        for i, target in enumerate(targets):
            error += 0.5 * (target - self.network_values.outputs[i]) ** 2
        return error

    def train(self, patterns, iterations=3000, learning_rate=0.0002):
        for i in range(iterations):
            error = 0.0
            for j, pattern in enumerate(patterns):
                inputs = pattern[0]
                targets = pattern[1]
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)
                if j % 50 == 0:
                    print 'error %-.5f' % error
            if i % 500 == 0:
                print 'error %-.5f' % error

    def predict(self, X):
        """
        :return: list of predictions after training algorithm
        """
        predictions = []
        for prediction in X:
            predictions.append(self.feed_forward(prediction))
        return predictions
