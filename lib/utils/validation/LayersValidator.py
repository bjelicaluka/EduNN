from lib.utils.validation.Validator import Validator


class LayersValidator(Validator):
    @staticmethod
    def validate(layers: list):
        error_msg = "Invalid layers architecture."
        if layers is None or len(layers) < 2:
            raise Exception(error_msg)
        for i in range(1, len(layers)):
            if len(layers[i]) != 2 or not isinstance(layers[i][0], int) or not isinstance(layers[i][1], str):
                raise Exception(error_msg)
