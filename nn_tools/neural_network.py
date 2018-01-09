import numpy as np

from nn_tools.nueron_tools import sigmoid, sigmoid_derivative


class NeuralNetwork(object):
    def __init__(self, input_neuron_count, hidden_neuron_count, output_neuron_count):
        self.input_neuron_count = input_neuron_count + 1  # Bias node addition
        self.hidden_neuron_count = hidden_neuron_count
        self.output_neuron_count = output_neuron_count
        # set up array of 1s for activations
        self.ai = [1.0] * self.input_neuron_count
        self.ah = [1.0] * self.hidden_neuron_count
        self.ao = [1.0] * self.output_neuron_count
        # create randomized weights
        self.wi = np.random.randn(self.input_neuron_count, self.hidden_neuron_count)
        self.wo = np.random.randn(self.hidden_neuron_count, self.output_neuron_count)
        # create arrays of 0 for changes
        self.ci = np.zeros((self.input_neuron_count, self.hidden_neuron_count))
        self.co = np.zeros((self.hidden_neuron_count, self.output_neuron_count))

    def feed_forward(self, inputs):
        if len(inputs) != self.input_neuron_count - 1:
            raise ValueError('Wrong number of inputs!')
            # input activations
        for i in range(self.input_neuron_count - 1):  # -1 is to avoid the bias
            self.ai[i] = inputs[i]
            # hidden activations
        for j in range(self.hidden_neuron_count):
            sum_hidden = 0.0
            for i in range(self.input_neuron_count):
                sum_hidden += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum_hidden)
            # output activations
        for k in range(self.output_neuron_count):
            sum_output = 0.0
            for j in range(self.hidden_neuron_count):
                sum_output += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum_output)
        return self.ao[:]

    def back_propagate(self, targets, learning_rate):
        """
        :param targets: y values
        :return: updated weights and current error
        """
        if len(targets) != self.output_neuron_count:
            raise ValueError('Wrong number of targets you silly goose!')
        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output_neuron_count
        for k in range(self.output_neuron_count):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = sigmoid_derivative(self.ao[k]) * error
        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden_neuron_count
        for j in range(self.hidden_neuron_count):
            error = 0.0
            for k in range(self.output_neuron_count):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = sigmoid_derivative(self.ah[j]) * error
        # update the weights connecting hidden to output
        for j in range(self.hidden_neuron_count):
            for k in range(self.output_neuron_count):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= learning_rate * change + self.co[j][k]
                self.co[j][k] = change
        # update the weights connecting input to hidden
        for i in range(self.input_neuron_count):
            for j in range(self.hidden_neuron_count):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= learning_rate * change + self.ci[i][j]
                self.ci[i][j] = change
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def train(self, patterns, iterations=3000, learning_rate=0.0002):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)
            if i % 500 == 0:
                print('error %-.5f' % error)

    def predict(self, X):
        """
        :return: list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feed_forward(p))
        return predictions
