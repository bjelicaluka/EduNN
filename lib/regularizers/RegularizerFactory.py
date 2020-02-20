from lib.regularizers.L2Regularizer import L2Regularizer


class RegularizerFactory(object):
    @staticmethod
    def get_regularizer(regularizer):
        if regularizer == "l2_regularizer":
            return L2Regularizer()
        else:
            raise Exception("Invalid regularizer name.")
