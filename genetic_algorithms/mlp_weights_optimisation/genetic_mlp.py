import numpy as np

from multilayer_perceptron.mlp.network import MLP
from multilayer_perceptron.mlp.activations import ActivationFunction
from multilayer_perceptron.mlp.lossfunc_and_metrics import LossFunction


class InitParamsMLP:
    __slots__ = ['input_size', 'hidden_layers_sizes', 'output_size',
                 'activation_func', 'out_func', 'loss']

    def __init__(self, input_size: int, hidden_layers_sizes: list[int], output_size: int,
                 activation_func: ActivationFunction, out_func: ActivationFunction,
                 loss: LossFunction):
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.activation_func = activation_func
        self.out_func = out_func
        self.loss = loss


class GeneticMLP:
    """A simple MLP which uses a genetic algorithm to find
    optimal weights for the network"""

    __slots__ = ['params_mlp', 'population', '__rng', 'init_population_size']

    def __init__(self, input_size: int, hidden_layers_sizes: list[int], output_size: int,
                 activation_func: ActivationFunction, out_func: ActivationFunction,
                 loss: LossFunction, population_size: int, random_state: int = None):
        """
        :param input_size:          dimension of the input
        :param hidden_layers_sizes: sizes of the hidden layers passed as a list (e.g. [10, 10])
        :param output_size:         dimension of the output
        :param activation_func:     activation function for all the layers except the last one
        :param out_func:            activation function for the last layer
        :param loss:                a loss function optimised during training
        :param population_size:     Initial size of the population
        :param random_state:        Controls the randomness of initial
                                    population and the training process
        """
        if random_state is not None:
            bit_generator = np.random.PCG64(random_state)
        else:
            bit_generator = np.random.PCG64()
        self.__rng = np.random.Generator(bit_generator)

        self.params_mlp = InitParamsMLP(
            input_size=input_size,
            hidden_layers_sizes=hidden_layers_sizes,
            output_size=output_size,
            activation_func=activation_func,
            out_func=out_func,
            loss=loss,
        )
        self.init_population_size = population_size
        self.population = [self.__new_mlp() for _ in range(population_size)]

    def __new_mlp(self, weights: list[np.ndarray] = None,
                  biases: list[np.ndarray] = None) -> MLP:
        mlp = MLP(
            input_size=self.params_mlp.input_size,
            hidden_layers_sizes=self.params_mlp.hidden_layers_sizes,
            output_size=self.params_mlp.output_size,
            activation_func=self.params_mlp.activation_func,
            out_func=self.params_mlp.out_func,
            loss=self.params_mlp.loss,
        )
        if weights is None or biases is None:
            mlp.set_default_weights(rng=self.__rng)
        else:
            mlp.weights = weights
            mlp.biases = biases
        return mlp

    @property
    def population_size(self) -> int:
        return len(self.population)

    def _crossover(self, ratio: float):
        """Performs a crossover on
        the specified ratio of the population"""
        crossover_count = int(np.ceil(self.population_size * ratio))
        if crossover_count < 2:  # too few individuals to perform a crossover
            return
        if crossover_count % 2 == 1:
            crossover_count -= 1  # it has to be an even number
        crossover_idx = self.__rng.choice(self.population_size, size=crossover_count, replace=False)

        for idx1, idx2 in zip(*np.split(crossover_idx, 2)):
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            child_weights = []
            child_biases = []
            for w1, b1, w2, b2 in zip(parent1.weights, parent1.biases,
                                      parent2.weights, parent2.biases):
                w_count = w1.size
                b_count = b1.size

                # number of weights to take from parent 1
                w_p1_count = self.__rng.integers(w_count)
                # flattened idx of weights to take from p1
                w_idx_p1 = self.__rng.choice(
                    np.arange(w_count), size=w_p1_count,
                )
                child_w = w2.flatten()
                child_w[w_idx_p1] = w1.flatten()[w_idx_p1]
                child_w = child_w.reshape(w1.shape)
                child_weights.append(child_w)

                # same deal with biases
                b_p1_count = self.__rng.integers(b_count)
                # flattened idx of weights to take from p1
                b_idx_p1 = self.__rng.choice(
                    np.arange(b_count), size=b_p1_count,
                )
                child_b = b2.flatten()
                child_b[b_idx_p1] = b1.flatten()[b_idx_p1]
                child_b = child_b.reshape(b1.shape)
                child_biases.append(child_b)

            child = self.__new_mlp(child_weights, child_biases)
            self.population.append(child)

    def _mutate(self, ratio: float):
        """Applies a gaussian mutation to the weights of
        the specified ratio of the population"""
        mutate_count = int(np.floor(self.population_size * ratio))
        mutate_idx = self.__rng.choice(self.population_size, size=mutate_count, replace=False)
        for idx in mutate_idx:
            mlp = self.population[idx]
            for i, w in enumerate(mlp.weights):
                mlp.weights[i] += self.__rng.normal(size=w.shape)
            for i, b in enumerate(mlp.biases):
                mlp.biases[i] += self.__rng.normal(size=b.shape)

    def _selection(self, x: np.ndarray, y: np.ndarray, tournament_size: int):
        """Performs a tournament selection"""
        new_population = []
        # new population will maintain the size of the initial population
        for _ in range(self.init_population_size):
            individuals_idx = self.__rng.choice(
                self.population_size, size=tournament_size, replace=False)
            individuals = [self.population[i] for i in individuals_idx]
            individuals_scores = [mlp.loss.val(y, mlp.predict(x)) for mlp in individuals]
            best_individual = individuals[np.argmin(individuals_scores)]
            # copy the best individual
            best_indiv_copy = self.__new_mlp(
                weights=best_individual.weights, biases=best_individual.biases)
            best_indiv_copy.set_normalisation(x, y)
            new_population.append(best_indiv_copy)

        self.population = new_population

    def predict(self, x: np.ndarray, y: np.ndarray,
                convert_prob_to_labels=False) -> np.ndarray:
        losses = [mlp.loss.val(y, mlp.predict(x)) for mlp in self.population]
        best_indiv_idx = np.argmin(losses)
        return self.population[best_indiv_idx].predict(x, convert_prob_to_labels)

    def loss_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculates loss function values for all individuals in the
        population and returns the lowest one"""
        losses = [mlp.loss.val(y, mlp.predict(x)) for mlp in self.population]
        return np.min(losses)

    def train(self, epochs: int,
              x_train: np.ndarray, y_train: np.ndarray,
              crossover_ratio: float = 0.7,
              mutation_ratio: float = 0.2,
              tournament_size: int = None,
              verbosity_period: int = 0):
        """
        Simulates the genetic algorithm

        :param x_train:             Training data
        :param y_train:             Target variable values
        :param epochs:              Number of iterations
        :param crossover_ratio:     Fraction of population to take part in a
                                    crossover
        :param mutation_ratio:      Fraction of population to mutate
        :param tournament_size:     Size of the selection tournament
        :param verbosity_period:    How often to print current loss function value
        """

        if tournament_size is None:
            tournament_size = int(np.ceil(self.population_size / 10))

        for epoch in range(1, epochs + 1):
            self._crossover(crossover_ratio)
            self._mutate(mutation_ratio)
            for mlp in self.population:
                mlp.set_normalisation(x_train, y_train)
            self._selection(x_train, y_train, tournament_size)
            if verbosity_period > 0 and epoch % verbosity_period == 0:
                print(f'Epoch {epoch} done! '
                      f'loss = {self.loss_value(x_train, y_train):.2f}')


