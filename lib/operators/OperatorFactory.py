from lib.operators.activation_functions.Sigmoid import Sigmoid
from lib.operators.activation_functions.Gaussian import Gaussian
from lib.operators.activation_functions.Tanh import Tanh
from lib.operators.activation_functions.ReLU import ReLU
from lib.operators.activation_functions.Sinusoid import Sinusoid
from lib.operators.activation_functions.SoftMax import SoftMax
from lib.operators.activation_functions.SoftPlus import SoftPlus


class OperatorFactory(object):
    @staticmethod
    def get_operator(operator):
        if operator == "sigmoid":
            return Sigmoid()
        elif operator == "sinusoid":
            return Sinusoid()
        elif operator == "relu":
            return ReLU()
        elif operator == "gaussian":
            return Gaussian()
        elif operator == "sinusoid":
            return Tanh()
        elif operator == "softmax":
            return SoftMax()
        elif operator == "softplus":
            return SoftPlus()
        else:
            raise Exception("Invalid operator name.")
