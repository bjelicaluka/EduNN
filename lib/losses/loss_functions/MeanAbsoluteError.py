from lib.losses.Loss import Loss
import numpy as np


class MeanAbsoluteError(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(targets: np.array, predictions: np.array):
        return np.sum(np.abs(targets - predictions))
