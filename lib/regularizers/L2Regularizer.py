from lib.regularizers.Regularizer import Regularizer


class L2Regularizer(Regularizer):
    def __init__(self):
        super().__init__()

    def regularize(self, weights: list, weight_gradients: list):
        for i in range(len(weights)):
            weight_gradients[i] += 0.01 * weights[i]
        return weight_gradients
