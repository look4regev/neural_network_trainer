from enum import Enum
import numpy as np


class ErrorCalculator:


#    define the cost functions we can use
    class CostFunction(Enum):
        QUADRATIC = 1
        CROSS_ENTROPY = 2

#    choose the cost function to use
    COST_FUNCTION = CostFunction.QUADRATIC


    @staticmethod
    def get_error(expected_vector, output_vector):
        if ErrorCalculator.COST_FUNCTION == ErrorCalculator.CostFunction.QUADRATIC:
            return ErrorCalculator.quadratic(expected_vector, output_vector)
        elif ErrorCalculator.COST_FUNCTION == ErrorCalculator.CostFunction.CROSS_ENTROPY:
            return ErrorCalculator.cross_entropy(expected_vector, output_vector)

    @staticmethod
    def quadratic(expected_vector, output_vector):
        error = 0.0
        for i, target in enumerate(expected_vector):
            error += 0.5 * (target - output_vector[i]) ** 2
        return error

#    https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
    @staticmethod
    def cross_entropy(expected_vector, output_vector):
        error = 0.0
        for i, target in enumerate(expected_vector):
            error += target * np.log(output_vector[i])
        return error * -1

