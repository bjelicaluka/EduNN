from abc import ABC, abstractmethod


class AbstractInitializer(ABC):
    @staticmethod
    @abstractmethod
    def init(layers: list, loss: str, learning_rate: float):
        pass
