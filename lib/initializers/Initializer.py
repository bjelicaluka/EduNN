import numpy as np
from lib.operators.OperatorFactory import OperatorFactory
from lib.losses.LossFactory import LossFactory
from lib.NeuralNetwork import NeuralNetwork
from lib.initializers.AbstractInitializer import AbstractInitializer


class Initializer(AbstractInitializer):
    @staticmethod
    def init(layers: list, loss: str = "mean_squared_error", learning_rate: float = 0.001):
        layer_sizes, operators = Initializer._generate_layers_and_operators(layers)
        weights, biases = Initializer._generate_weights_and_biases(layer_sizes)
        loss_function = LossFactory.get_loss(loss)
        return NeuralNetwork(weights, biases, layer_sizes, operators, loss_function, learning_rate)

    @staticmethod
    def _generate_layers_and_operators(layers: list):
        layer_sizes = []
        operators = []
        for i in range(len(layers)):
            layer = layers[i]
            layer_size = layer[0]
            layer_sizes.append(layer_size)
            if i != 0:
                operator_string = layer[1]
                operator = OperatorFactory.get_operator(operator_string)
                operators.append(operator)
        return layer_sizes, operators

    @staticmethod
    def _generate_weights_and_biases(layer_sizes):
        weights = []
        biases = []
        for i in range(1, len(layer_sizes)):
            previous_num = layer_sizes[i - 1]
            current_num = layer_sizes[i]
            biases.insert(i - 1, np.zeros((1, current_num)))
            weights.insert(i - 1, np.random.normal(0, 1, [previous_num, current_num]))
        return weights, biases
