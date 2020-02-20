from lib.operators.Operator import Operator
import numpy as np


class ReLU (Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x):
        return x
