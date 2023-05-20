from typing import Callable

import numpy as np


class BaseGenetic:
    """A basic genetic algorithm for mathematical functions optimisation"""

    __slots__ = ['population', 'loss_fn', '__rng', 'init_population_size']

    def __init__(self, input_dim: int,
                 population_size: int,
                 loss_fn: Callable[[np.ndarray], int],
                 init_interval_radius: float = 1,
                 random_state: int = None):
        """
        :param input_dim:               Dimension of the input data
        :param population_size:         Initial size of the population
        :param loss_fn:                 A function to optimise
        :param init_interval_radius:    Controls the size of the interval from
                                        which the initial population is drawn.
                                        For example, if set to x, initial
                                        population will be drawn from a uniform
                                        distribution on [-x, x]
        :param random_state:            Controls the randomness of initial
                                        population and the training process
        """
        if random_state is not None:
            bit_generator = np.random.PCG64(random_state)
        else:
            bit_generator = np.random.PCG64()
        self.__rng = np.random.Generator(bit_generator)

        self.init_population_size = population_size
        self.population = self.__rng.uniform(
            low=-init_interval_radius, high=init_interval_radius,
            size=(population_size, input_dim)
        )
        self.loss_fn = loss_fn

    @property
    def population_size(self) -> int:
        return self.population.shape[0]

    @property
    def input_dim(self) -> int:
        return self.population.shape[1]

    def _crossover(self, ratio: float):
        """Performs a single-point crossover on
        the specified ratio of the population"""
        crossover_count = int(np.ceil(self.population_size * ratio))
        if crossover_count < 2:  # too few individuals to perform a crossover
            return
        if crossover_count % 2 == 1:
            crossover_count -= 1  # it has to be an even number
        crossover_idx = self.__rng.choice(self.population_size, size=crossover_count, replace=False)

        for idx1, idx2 in zip(*np.split(crossover_idx, 2)):
            crossover_point = self.__rng.integers(1, self.input_dim - 1)
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            child = np.concatenate(
                [parent1[:crossover_point],
                 parent2[crossover_point:]]
            )
            self.population = np.concatenate([self.population, child.reshape(1, -1)])

    def _mutate(self, ratio: float):
        """Applies a gaussian mutation to
        the specified ratio of the population"""
        mutate_count = int(np.floor(self.population_size * ratio))
        mutate_idx = self.__rng.choice(self.population_size, size=mutate_count, replace=False)
        mutation = self.__rng.normal(size=(mutate_count, self.input_dim))
        self.population[mutate_idx] += mutation

    def _selection(self, tournament_size: int):
        """Performs a tournament selection"""
        new_population_idx = []
        # new population will maintain the size of the initial population
        for _ in range(self.init_population_size):
            individuals_idx = self.__rng.choice(
                self.population_size, size=tournament_size, replace=False)
            individuals_scores = [self.loss_fn(i) for i in self.population[individuals_idx]]
            best_individual_idx = individuals_idx[np.argmin(individuals_scores)]
            new_population_idx.append(best_individual_idx)

        self.population = self.population[new_population_idx]

    def loss_value(self) -> float:
        """Calculates loss function values for all individuals in the
        population and returns the lowest one"""
        losses = [self.loss_fn(individual) for individual in self.population]
        return np.min(losses)

    def train(self, epochs: int,
              crossover_ratio: float = 0.7,
              mutation_ratio: float = 0.2,
              tournament_size: int = None,
              verbosity_period: int = 0):
        """
        Simulates the genetic algorithm

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
            self._selection(tournament_size)
            if verbosity_period > 0 and epoch % verbosity_period == 0:
                print(f'Epoch {epoch} done! Current loss value: {self.loss_value()}')
