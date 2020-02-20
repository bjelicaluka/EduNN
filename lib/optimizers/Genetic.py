from lib.initializers.Initializer import Initializer
from lib.optimizers.Optimizer import Optimizer
from lib.NeuralNetwork import NeuralNetwork
from lib.utils.utility_functions import count_decimals
import numpy as np
from random import randint, uniform


class Genetic(Optimizer):
    def __init__(self, layers: list, loss: str = "mean_square_error", learning_rate: float = 0.001):
        super().__init__(layers, loss, learning_rate)
        self._layers = layers
        self._loss = loss
        self._learning_rate = learning_rate
        self._population = []
        self._scores = []
        self._population_size = 20
        self._mutation_probability = 0.01
        self._elitism_num = 3
        self._generate_population()

    def set_population_size(self, population_size: int):
        self._population_size = population_size

    def set_elitism_num(self, elitism_num: int):
        self._elitism_num = elitism_num

    def set_mutation_probability(self, mutation_probability: float,):
        self._mutation_probability = mutation_probability

    def train(self, inputs: list, labels: list, epochs: int, print_epochs: bool = True):
        for _ in range(epochs):
            self._calculate_scores(inputs=inputs, labels=labels)
            self._elitism()
            self._calculate_probabilities()
            self._crossover()
            self._mutation()

            self._generate_population()
            for i in range(self._population_size):
                nn = self._population[i]
                nn.set_weights_and_biases(self._children[i][0], self._children[i][1])
            if print_epochs:
                best_score = self._scores[int(np.argmax(self._scores))]
                print(f"=== Epoch {_}/{epochs}\tBest Score: {best_score} ===")

        return self._population[int(np.argmax(self._scores))]

    def _calculate_scores(self, inputs: list, labels: list):
        self._scores = []
        self._score_sum = 0
        for nn in self._population:
            score = self._calculate_fitness(nn=nn, inputs_=inputs, labels_=labels)
            self._scores.append(score)
            self._score_sum += score

    @staticmethod
    def _calculate_fitness(nn: NeuralNetwork, inputs_: list, labels_: list):
        scored = 0
        for j in range(len(inputs_)):
            inputs = inputs_[j]
            data_label = labels_[j]
            prediction = nn.predict(inputs)
            predicted_index = np.argmax(prediction)
            label_index = np.argmax(data_label)
            if predicted_index == label_index:
                scored += 1
        return scored / len(inputs_)

    def _elitism(self):
        self._children = []

        self._best_scores = sorted(range(len(self._scores)), key=lambda sub: self._scores[sub])[-self._elitism_num:]
        for i in range(len(self._population)):
            if i in self._best_scores:
                weights, biases = self._population[i].get_weights_and_biases()
                self._children.append([weights, biases])
        self._remove_units()

    def _remove_units(self):
        new_population = []
        new_scores = []
        for i in range(self._population_size):
            if i not in self._best_scores:
                new_population.append(self._population[i])
                new_scores.append(self._scores[i])
            else:
                self._score_sum -= self._scores[i]
        self._population = new_population
        self._scores = new_scores

    def _calculate_probabilities(self):
        for i in range(len(self._scores)):
            self._scores[i] /= self._score_sum

    def _crossover(self):
        for _ in range(self._population_size - self._elitism_num):
            parents = self._selection()
            weights = []
            biases = []
            weights1, biases1 = self._population[parents[0]].get_weights_and_biases()
            weights2, biases2 = self._population[parents[1]].get_weights_and_biases()

            for i in range(len(biases1)):
                r = randint(0, 1)
                b = []
                for j in range(len(biases1[i])):
                    if r == 0:
                        b.append(biases1[i][j])
                    else:
                        b.append(biases2[i][j])
                biases.append(b)

            for i in range(len(weights1)):
                w = []
                for j in range(len(weights1[i])):
                    ww = []
                    for k in range(len(weights1[i][j])):
                        r = randint(0, 1)
                        if r == 0:
                            ww.append(weights1[i][j][k])
                        else:
                            ww.append(weights2[i][j][k])
                    w.append(ww)
                weights.append(w)

            self._children.append([weights, biases])

    def _selection(self):
        return np.random.choice(len(self._population), 2, p=self._scores)

    def _mutation(self):
        for child in self._children:
            weights = child[0]
            biases = child[1]
            for i in range(len(weights)):
                for j in range(len(biases[i])):
                    if self._mutate_gen():
                        child[1][i][j] *= uniform(0.01, 0.5)
                for j in range(len(weights[i])):
                    if self._mutate_gen():
                        index = randint(0, len(weights[i][j]) - 1)
                        child[0][i][j][index] *= uniform(0.01, 0.5)

    def _mutate_gen(self):
        precision = count_decimals(self._mutation_probability)
        r = randint(0, precision) / precision
        return r <= self._mutation_probability

    def _generate_population(self):
        self._population = []
        for _ in range(self._population_size):
            nn = Initializer.init(self._layers, self._loss, self._learning_rate)
            self._population.append(nn)
