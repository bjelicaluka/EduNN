from lib.losses.Loss import Loss
import numpy as np


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(targets: np.array, predictions: np.array):
        indices = np.argmax(predictions, axis=1).astype(int)
        probability = targets[np.arange(len(targets)), indices]
        log = np.log(probability)
        loss = -1.0 * np.sum(log) / len(log)
        return loss
