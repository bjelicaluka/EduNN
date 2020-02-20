from lib.NeuralNetwork import NeuralNetwork
from lib.initializers.Initializer import Initializer
from lib.utils.validation.LayersValidator import LayersValidator
from lib.utils.validation.DataValidator import DataValidator
from lib.optimizers.OptimizerFactory import OptimizerFactory
from lib.utils.evaluation_metrics.EvaluationMetricFactory import EvaluationMetricFactory
import numpy as np
from pickle import load


class Model(object):
    def __init__(self, layers: list = None, loss_function: str = "mean_squared_error", learning_rate: float = 0.001,
                 neural_network: NeuralNetwork = None):
        self._learning_rate = learning_rate
        self._loss_function = loss_function
        self._layers = layers
        if neural_network is None:
            LayersValidator.validate(layers)
            self._nn = Initializer.init(layers, loss_function, learning_rate)
        else:
            self._nn = neural_network

    @classmethod
    def from_network(cls, nn):
        loss = nn.get_loss_function()
        lr = nn.get_learning_rate()
        return cls(None, loss, lr, nn)

    @classmethod
    def from_file(cls, file_path: str):
        if file_path is None:
            raise Exception("Invalid file path.")
        nn = load(open(file_path, "rb"))
        return Model.from_network(nn)

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, learning_rate: float):
        self._nn.set_learning_rate(learning_rate)
        self._learning_rate = learning_rate

    def get_loss_function(self):
        return self._loss_function

    def set_loss_function(self, loss_function: str):
        self._nn.set_loss_function(loss_function)
        self._loss_function = loss_function

    def calculate_accuracy(self, inputs=None, labels=None, report_type: str = "classification_report"):
        DataValidator.validate(inputs)
        evaluation_metric_factory = EvaluationMetricFactory(self._nn)
        evaluation_metric = evaluation_metric_factory.get_evaluation_metric(report_type)
        return evaluation_metric.evaluate(inputs, labels)

    def train(self, optimization_algorithm: str = "gradient_descent", inputs: np.array = None, labels: np.array = None,
              epochs: int = 100, shuffle_data: bool = False, population_size: int = 20,
              mutation_probability: float = 0.01, elitism_num: int = 3, batch_size: int = 10,
              print_epochs: bool = True):
        DataValidator.validate(inputs)
        optimizer_factory = OptimizerFactory(self._layers, self._loss_function, self._learning_rate, shuffle_data,
                                             population_size, mutation_probability, elitism_num, batch_size,
                                             print_epochs)
        optimizer = optimizer_factory.get_optimizer(optimization_algorithm)
        self._nn = optimizer.train(np.array(inputs), np.array(labels), epochs, print_epochs)

    def predict(self, input_array: np.array):
        return self._nn.predict(input_array)

    def save(self, path: str, file_name: str):
        self._nn.save(path, file_name)
