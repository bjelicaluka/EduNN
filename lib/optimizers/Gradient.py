import numpy as np
from lib.utils.utility_functions import shuffle_data
from lib.operators.activation_functions.ReLU import ReLU
from lib.initializers.Initializer import Initializer
from lib.optimizers.Optimizer import Optimizer
from lib.regularizers.RegularizerFactory import RegularizerFactory


class Gradient(Optimizer):
    def __init__(self, layers: list, loss: str = "mean_square_error", learning_rate: float = 0.001):
        super().__init__(layers, loss, learning_rate)
        self._nn = Initializer.init(layers, loss, learning_rate)
        self._loss_function = self._nn.get_loss_function()
        self._learning_rate = self._nn.get_learning_rate()
        self._batch_size = 1
        self._do_shuffle_data = False
        self._weights, self._biases = self._nn.get_weights_and_biases()

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def set_shuffle_data(self, do_shuffle_data: bool):
        self._do_shuffle_data = do_shuffle_data

    def train(self, inputs, labels, epochs: int, print_epochs: bool = True):
        for epoch in range(epochs):
            inputs, labels = self._shuffle_data(inputs, labels)

            iteration = 0
            while iteration < len(inputs):
                inputs_batch, labels_batch = self._create_mini_batch(iteration, inputs, labels)

                layers = self._nn.feed_forward(inputs_batch)

                loss = self._loss_function.function(layers[len(layers) - 1], labels_batch)

                deltas = self._backward_pass(layers, labels_batch)

                w_gradients, b_gradients = self._backpropagation(inputs_batch, deltas, layers)

                w_gradients = self._regularization(w_gradients)

                self._gradient_descent(layers, w_gradients, b_gradients)

                if print_epochs and iteration % 10000 == 0:
                    print(f"=== Epoch: {epoch+1}/{epochs}\tIteration:{iteration}\tLoss: {loss} ===")

                iteration += self._batch_size
        self._update_network()
        return self._nn

    def _shuffle_data(self, inputs, labels):
        if self._do_shuffle_data:
            return shuffle_data(inputs, labels)

    def _create_mini_batch(self, iteration, inputs, labels):
        inputs_batch = np.array(inputs[iteration:iteration + self._batch_size])
        labels_batch = np.array(labels[iteration:iteration + self._batch_size])
        return inputs_batch, labels_batch

    def _backward_pass(self, layers, labels_batch):
        delta = None
        deltas = []
        for i in range(len(layers) - 1, -1, -1):
            is_output_layer = i == len(layers) - 1
            if is_output_layer:
                delta = (layers[i] - labels_batch) / self._batch_size
            else:
                delta = np.dot(delta, self._weights[i + 1].T)
                delta = self._nn.get_operator(i).derivative(delta)
            deltas.append(delta)
        return deltas

    @classmethod
    def _backpropagation(cls, inputs_batch, deltas, layers):
        w_grad = np.dot(inputs_batch.T, deltas[len(deltas) - 1])  # forward * backward
        b_grad = np.sum(deltas[len(deltas) - 1], axis=0, keepdims=True)
        w_gradients = [w_grad]
        b_gradients = [b_grad]
        k = 0
        for i in range(len(layers) - 2, -1, -1):
            w_grad = np.dot(layers[k].T, deltas[i])  # forward * backward
            w_gradients.append(w_grad)
            b_grad = np.sum(deltas[i], axis=0, keepdims=True)
            b_gradients.append(b_grad)
            k += 1
        return w_gradients, b_gradients

    def _regularization(self, weight_gradients):
        regularizer = RegularizerFactory.get_regularizer("l2_regularizer")
        return regularizer.regularize(self._weights, weight_gradients)

    def _gradient_descent(self, layers, weight_gradients, bias_gradients):
        for i in range(len(layers) - 1, -1, -1):
            self._weights[i] -= self._learning_rate * weight_gradients[i]
            self._biases[i] -= self._learning_rate * bias_gradients[i]

    def _update_network(self):
        self._nn.set_weights_and_biases(self._weights, self._biases)
