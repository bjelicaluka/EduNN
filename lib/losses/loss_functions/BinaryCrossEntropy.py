from lib.losses.Loss import Loss
import numpy as np


class BinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(targets: np.array, predictions: np.array):
        n = predictions.shape[0]
        ce = -np.sum(targets * np.log(predictions)) / n
        return ce
