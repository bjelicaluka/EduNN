from lib.operators.Operator import Operator
import numpy as np


class Tanh(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def derivative(x):
        return 1 - np.power(Tanh.function(x), 2)
