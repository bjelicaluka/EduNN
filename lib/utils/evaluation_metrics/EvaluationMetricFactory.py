from lib.utils.evaluation_metrics.AverageSumErrorEvaluationMetric import AverageSumErrorEvaluationMetric
from lib.utils.evaluation_metrics.SumErrorEvaluationMetric import SumErrorEvaluationMetric
from lib.utils.evaluation_metrics.ClassificationEvaluationMetric import ClassificationEvaluationMetric
from lib.NeuralNetwork import NeuralNetwork


class EvaluationMetricFactory(object):
    def __init__(self, nn: NeuralNetwork):
        self._nn = nn

    def get_evaluation_metric(self, evaluation_metric: str):
        if evaluation_metric == "classification_report":
            return ClassificationEvaluationMetric(self._nn)
        elif evaluation_metric == "average_sum_error_report":
            return AverageSumErrorEvaluationMetric(self._nn)
        elif evaluation_metric == "sum_error_report":
            return SumErrorEvaluationMetric(self._nn)
        else:
            raise Exception("Invalid report type.")
