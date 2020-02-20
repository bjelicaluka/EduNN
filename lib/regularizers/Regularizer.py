from abc import ABC, abstractmethod


class Regularizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def regularize(self, weights: list, weight_gradients: list):
        pass
