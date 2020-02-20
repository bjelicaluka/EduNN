from lib.operators.Operator import Operator
import numpy as np


class Sinusoid(Operator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def function(x):
        return np.sin(x)

    @staticmethod
    def derivative(x):
        return np.cos(x)
