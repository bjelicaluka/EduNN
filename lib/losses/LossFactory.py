from lib.losses.loss_functions.MeanAbsoluteError import MeanAbsoluteError
from lib.losses.loss_functions.MeanSquaredError import MeanSquaredError
from lib.losses.loss_functions.BinaryCrossEntropy import BinaryCrossEntropy
from lib.losses.loss_functions.CrossEntropy import CrossEntropy


class LossFactory(object):
    @staticmethod
    def get_loss(loss_function):
        if loss_function == "binary_cross_entropy":
            return BinaryCrossEntropy()
        elif loss_function == "cross_entropy":
            return CrossEntropy()
        elif loss_function == "mean_squared_error":
            return MeanSquaredError()
        elif loss_function == "mean_absolute_error":
            return MeanAbsoluteError()
        else:
            raise Exception("Invalid loss function name.")
