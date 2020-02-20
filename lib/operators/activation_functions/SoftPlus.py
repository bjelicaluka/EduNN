from lib.operators.Operator import Operator
import numpy as np


class SoftPlus(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def derivative(x):
        return 1 / 1 + np.exp(-x)
