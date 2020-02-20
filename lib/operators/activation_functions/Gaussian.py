from lib.operators.Operator import Operator
import numpy as np


class Gaussian(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return np.exp(-(np.power(x, 2)))

    @staticmethod
    def derivative(x):
        return -2 * x * np.exp(-(np.power(x, 2)))
