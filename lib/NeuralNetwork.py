import numpy as np
from pickle import dump
from lib.losses.Loss import Loss
from lib.losses.LossFactory import LossFactory
import os


class NeuralNetwork(object):
    def __init__(self, weights: list, biases: list, layers: list, operators: list,
                 loss_function: Loss, learning_rate: float = 0.001):
        self._learning_rate = learning_rate
        self._loss_function = loss_function
        self._weights = weights
        self._biases = biases
        self._layer_sizes = layers
        self._operators = operators

    def predict(self, input_array: np.array):
        layers = self._feed_layer_sizes(input_array)
        outputs = layers[len(layers) - 1]
        return outputs

    def feed_forward(self, inputs_batch: np.array):
        return self._feed_layer_sizes(inputs_batch)

    def _feed_layer_sizes(self, inputs_batch: np.array):
        layers = []
        activated = None
        for i in range(len(self._layer_sizes) - 1):
            if i == 0:
                batch = inputs_batch
            else:
                batch = activated
            activation = self._operators[i].function
            z = np.dot(batch, self._weights[i]) + self._biases[i]
            activated = activation(z)
            layers.append(activated)
        return layers

    def get_operator(self, index):
        return self._operators[index]

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_loss_function(self):
        return self._loss_function

    def set_loss_function(self, loss_function: str):
        self._loss_function = LossFactory.get_loss(loss_function)

    def get_weights_and_biases(self):
        return self._weights, self._biases

    def set_weights_and_biases(self, weights: list, biases: list):
        self._weights = weights
        self._biases = biases

    def get_layer_sizes(self):
        layers = [l for l in self._layer_sizes]
        return layers

    def print_weights(self):
        for w in self._weights:
            print(w)

    def print_biases(self):
        for b in self._biases:
            print(b)

    def save(self, path: str, file_name: str):
        if not os.path.exists(path):
            os.makedirs(path)
        dump(self, open(path+file_name, "wb"))
