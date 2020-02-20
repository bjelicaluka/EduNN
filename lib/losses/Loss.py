from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def function(targets: np.array, predictions: np.array):
        pass
