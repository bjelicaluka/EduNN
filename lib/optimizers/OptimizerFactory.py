from lib.optimizers.Genetic import Genetic
from lib.optimizers.Gradient import Gradient


class OptimizerFactory(object):
    def __init__(self, layers: list, loss_function: str, learning_rate: float, shuffle_data: bool, population_size: int,
                 mutation_probability: float, elitism_num: int, batch_size: int, print_epochs: bool = True):
        self._layers = layers
        self._loss_function = loss_function
        self._learning_rate = learning_rate
        self._shuffle_data = shuffle_data
        self._population_size = population_size
        self._mutation_probability = mutation_probability
        self._elitism_num = elitism_num
        self._batch_size = batch_size
        self._print_epochs = print_epochs

    def get_optimizer(self, optimizer):
        if optimizer == "gradient_descent":
            g = Gradient(self._layers, self._loss_function, self._learning_rate)
            g.set_shuffle_data(self._shuffle_data)
            g.set_batch_size(self._batch_size)
            return g
        elif optimizer == "genetic_algorithm":
            g = Genetic(self._layers, self._loss_function, self._learning_rate)
            g.set_population_size(self._population_size)
            g.set_mutation_probability(self._mutation_probability)
            g.set_elitism_num(self._elitism_num)
            return g
        else:
            raise Exception("Invalid optimizer name.")
