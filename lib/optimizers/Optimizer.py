from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, layers: list, loss: str = "mean_square_error", learning_rate: float = 0.001):
        pass

    @abstractmethod
    def train(self, inputs, labels, epochs: int, print_epochs: bool = True):
        pass
