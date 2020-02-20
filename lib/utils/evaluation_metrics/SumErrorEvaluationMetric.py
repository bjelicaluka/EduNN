import numpy as np
from lib.NeuralNetwork import NeuralNetwork
from lib.utils.evaluation_metrics.EvaluationMetric import EvaluationMetric


class SumErrorEvaluationMetric(EvaluationMetric):
    def __init__(self, nn: NeuralNetwork):
        super().__init__(nn)
        self._nn = nn

    def evaluate(self, inputs: list, labels: list):
        errors = []
        for j in range(0, len(inputs)):
            input_example = inputs[j]
            data_label = labels[j]
            nn_output = self._nn.predict(input_example)
            errors.append(abs(np.sum(data_label - nn_output)))
        return np.average(errors)
