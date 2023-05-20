from typing import Literal

import numpy as np
import pandas as pd

from core import Rectangle, Circle


class CuttingStockGenetic:
    """A genetic algorithm for solving a cutting stock problem"""

    __slots__ = ['population', '__rng', 'init_population_size', 'rectangles_unique',
                 'circle_radius']

    def __init__(self, circle_radius: float,
                 rectangles_data: pd.DataFrame,
                 population_size: int,
                 random_state: int = None):
        """
        :param circle_radius:           Radius of the circle in the problem
        :param rectangles_data:         A data frame with data about rectangles.
                                        Required columns: "height", "width", "value"
        :param population_size:         Initial size of the population
        :param random_state:            Controls the randomness of initial
                                        population and the training process
        """
        if random_state is not None:
            bit_generator = np.random.PCG64(random_state)
        else:
            bit_generator = np.random.PCG64()
        self.__rng = np.random.Generator(bit_generator)

        self.init_population_size = population_size
        self.circle_radius = circle_radius

        self.rectangles_unique = self.__unique_rectangles_from(rectangles_data)

        self.population = [Circle(circle_radius) for _ in range(population_size)]
        self.__reset_population()

    @property
    def population_size(self) -> int:
        return len(self.population)

    @staticmethod
    def __unique_rectangles_from(rectangles_data: pd.DataFrame) -> list[Rectangle]:
        rect_unique = []

        def add_rectangles_from_row(df_row):
            rect = Rectangle(
                value=df_row['value'],
                height=df_row['height'], width=df_row['width']
            )
            rect_unique.append(rect)
            # add a flipped version of rectangle as well if possible
            if rect.height != rect.width:
                rect_flipped = Rectangle(
                    value=df_row['value'],
                    height=df_row['width'], width=df_row['height']
                )
                rect_unique.append(rect_flipped)

        rectangles_data.apply(add_rectangles_from_row, axis=1)
        return rect_unique

    def __reset_population(self):
        self.population = []
        for _ in range(self.init_population_size):
            new_circle = Circle(self.circle_radius)
            new_circle.rows.append(self.__rng.choice(self.rectangles_unique))
            # add new rows until there is overflow
            while not new_circle.is_overflow():
                new_circle.rows.append(self.__rng.choice(self.rectangles_unique))
            # remove overflow
            new_circle.trim()
            self.population.append(new_circle)

    def _crossover(self, ratio: float):
        """Performs a single-point crossover on
        the specified ratio of the population"""
        crossover_count = int(np.ceil(self.population_size * ratio))
        if crossover_count < 2:  # too few individuals to perform a crossover
            return
        if crossover_count % 2 == 1:
            crossover_count -= 1  # it has to be an even number
        crossover_idx = self.__rng.choice(
            self.population_size, size=crossover_count, replace=False)

        for idx1, idx2 in zip(*np.split(crossover_idx, 2)):
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            # parent1 should be the one with more rows
            if len(parent1.rows) < len(parent2.rows):
                parent1, parent2 = parent2, parent1
            # if parent1 has too few rows, the crossover is canceled
            if len(parent1.rows) < 3:
                continue

            child = Circle(self.circle_radius)
            crossover_point_p1 = self.__rng.integers(1, len(parent1.rows) - 1)
            child.rows += parent1.rows[:crossover_point_p1]
            # add rows from the top of the second parent until there is overflow
            for i in range(-1, -len(parent2.rows) - 1, -1):
                child.rows.insert(crossover_point_p1, parent2.rows[i])
                # stop if overflow
                if child.is_overflow():
                    # remove the last added row (that caused an overflow)
                    del child.rows[crossover_point_p1]
                    break
            # add a child to the population
            self.population.append(child)

    def _mutate(self, ratio: float):
        """Applies a mutation to the specified ratio of the population.
        Mutation picks a random row and then randomly changes its
        rectangle type. If there is an overflow, the circle is trimmed.
        If not, new rectangles will be added to the top as long as there
        is no overflow."""
        mutate_count = int(np.floor(self.population_size * ratio))
        mutate_idx = self.__rng.choice(
            self.population_size, size=mutate_count, replace=False)
        for idx in mutate_idx:
            circle = self.population[idx]
            row_to_alter = self.__rng.integers(len(circle.rows))
            new_rect = self.__rng.choice(self.rectangles_unique)
            circle.rows[row_to_alter] = new_rect
            while not circle.is_overflow():
                circle.rows.append(self.__rng.choice(self.rectangles_unique))
            circle.trim()

    def _selection(self, tournament_ratio: float):
        """Performs a tournament selection"""
        tournament_size = int(np.ceil(self.population_size * tournament_ratio))
        new_population = []
        # new population will maintain the size of the initial population
        while len(new_population) < self.init_population_size:
            individuals_idx = self.__rng.choice(
                self.population_size, size=tournament_size, replace=False)
            individuals_scores = [self.population[idx].score() for idx in individuals_idx]
            best_individual_idx = individuals_idx[np.argmax(
                individuals_scores)]
            new_population.append(self.population[best_individual_idx])

        self.population = new_population

    def best_individual(self) -> Circle:
        """Returns the circle with the highest score"""
        scores = [c.score() for c in self.population]
        return self.population[np.argmax(scores)]

    def score(self) -> float:
        """Score of the best individual"""
        return self.best_individual().score()

    def train(self, epochs: int,
              crossover_ratio: float = 0.7,
              mutation_ratio: float = 0.2,
              tournament_ratio: float = 0.1,
              verbosity_period: int = 0):
        """
        Simulates the genetic algorithm

        :param epochs:                  Number of iterations
        :param crossover_ratio:         Fraction of population to take part in a
                                        crossover
        :param mutation_ratio:          Fraction of population to mutate
        :param tournament_ratio:        Ratio of population to take part in a
                                        single tournament
        :param verbosity_period:        How often to print current score
        """
        for epoch in range(1, epochs + 1):
            self._crossover(crossover_ratio)
            self._mutate(mutation_ratio)
            self._selection(tournament_ratio)
            if verbosity_period > 0 and epoch % verbosity_period == 0:
                print(f'Epoch {epoch} done! Current score: {self.score()}')
