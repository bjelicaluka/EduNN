import numpy as np
from sklearn.metrics.classification import classification_report
from lib.utils.evaluation_metrics.EvaluationMetric import EvaluationMetric
from lib.NeuralNetwork import NeuralNetwork


class ClassificationEvaluationMetric(EvaluationMetric):
    def __init__(self, nn: NeuralNetwork):
        super().__init__(nn)
        self._nn = nn

    def evaluate(self, inputs: list, labels: list):
        predictions = []
        labels_max = []
        for j in range(0, len(inputs)):
            input_excample = inputs[j]
            data_label = labels[j]
            nn_output = self._nn.predict(input_excample)
            predicted_index = np.argmax(nn_output)
            predictions.append(predicted_index)
            label_index = np.argmax(data_label)
            labels_max.append(label_index)
        return classification_report(labels_max, predictions, output_dict=False)
