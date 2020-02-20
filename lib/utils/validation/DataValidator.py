import numpy as np
from lib.utils.validation.Validator import Validator


class DataValidator(Validator):
    @staticmethod
    def validate(data: list):
        error_msg = "Training data is not valid. Please ensure that training data " \
                    "is specified in form of: [num1, num2, ...]"
        if data is None:
            raise Exception(error_msg)
        if not isinstance(data, (np.ndarray, list)):
            raise Exception(error_msg)
        if len(data) == 0:
            raise Exception("Cannot train model with empty dataset.")
        for d in data:
            if not isinstance(d, (int, float, np.ndarray)):
                raise Exception(error_msg)
