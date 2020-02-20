from lib.losses.Loss import Loss
import numpy as np


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(targets: np.array, predictions: np.array):
        return np.sum(np.power((targets - predictions), 2))
