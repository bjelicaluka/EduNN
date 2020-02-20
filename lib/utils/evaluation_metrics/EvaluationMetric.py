from abc import ABC, abstractmethod
from lib.NeuralNetwork import NeuralNetwork


class EvaluationMetric(ABC):
    @abstractmethod
    def __init__(self, nn: NeuralNetwork):
        pass

    @abstractmethod
    def evaluate(self, inputs: list, labels: list):
        pass
