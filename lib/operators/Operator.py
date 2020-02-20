from abc import ABC, abstractmethod


class Operator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def function(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass
