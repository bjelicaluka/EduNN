from lib.operators.Operator import Operator
import numpy as np


class SoftMax(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def derivative(x):
        return x * (1. - x)
