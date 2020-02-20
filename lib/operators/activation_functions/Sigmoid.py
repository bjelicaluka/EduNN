from lib.operators.Operator import Operator
import numpy as np


class Sigmoid(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return np.multiply(x, (1 - x))
