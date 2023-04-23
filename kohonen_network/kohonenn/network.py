from typing import Callable

import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt

from .neighbourhood_func import NeighbourhoodFunction


class KohonenNetwork:
    __slots__ = ['input_size', 'output_dim', 'weights', '__rng',
                 '__normalise_min', '__normalise_max']

    def __init__(self, input_size: int, output_dim: tuple[int, int]):
        self.input_size = input_size
        self.output_dim = output_dim
        self.__reset_weights()

    def __normalise_input(self, data: np.ndarray, reset_norm_coefs=False) -> np.ndarray:
        if reset_norm_coefs:
            self.__normalise_max = np.max(data)
            self.__normalise_min = np.min(data)
        return (data - self.__normalise_min) / (self.__normalise_max - self.__normalise_min)

    def __reset_weights(self, low: float = 0, high: float = 1, random_state: int = None):
        """
        self.weights[:, :, i] - weights for the i-th input
        self.weights[i][j] - weights for the (i, j)-th neuron
        """
        if random_state is None:
            rng = Generator(PCG64())
        else:
            rng = Generator(PCG64(random_state))
        self.__rng = rng

        self.weights = self.__rng.uniform(
            low=low, high=high,
            size=(self.output_dim[0], self.output_dim[1], self.input_size)
        )

    def __shuffled_vector(self, v: np.ndarray) -> np.ndarray:
        """Returns a shuffled copy of a given vector"""
        indices = np.arange(v.shape[0])
        self.__rng.shuffle(indices)
        return v[indices]

    def visualise_weights(self, size: int = None) -> plt.Figure | None:
        if self.input_size == 2:
            x = self.weights[:, :, 0].reshape(1, -1)
            y = self.weights[:, :, 1].reshape(1, -1)
            fig = plt.figure()
            plt.scatter(x, y, s=size)
            plt.xlabel('x1 weights')
            plt.ylabel('x2 weights')
            plt.title('Neuron weights')
            return fig
        print('Currently only supported for 2-dimensional input ;(')
        return

    def train(self, data: np.ndarray, epochs: int, init_lr: float, lr_decay_func: Callable,
              neighbourhood_func: NeighbourhoodFunction, verbosity_period: int = 0,
              random_state: int = None):
        """
        Attemps to find optimal weights for the network

        :param data:                training data without cluster labels
        :param epochs:              number of training iterations
        :param init_lr:             initial learning rate value
        :param lr_decay_func:       function used to compute the learning rate based on the current
                                    epoch number and the total number of epochs
        :param neighbourhood_func:  neighbourhood weight function
        :param verbosity_period:    controls how often current epoch numbers are printed. If
                                    verbosity_period = n, then the progress will be printed
                                    every nth epoch. If verbosity_period = 0, then nothing
                                    will be printed
        :param random_state:        controls the randomness of initial weights
        """
        self.__reset_weights(np.min(data), np.max(data), random_state)

        # distance function
        dist = neighbourhood_func.distance_func

        for epoch in range(epochs):
            # epochs start from 1
            epoch += 1
            deltas_abs = []

            for x in self.__shuffled_vector(data):
                weights_dists = [dist(self.weights[i][j], x)
                                 for i in range(self.output_dim[0])
                                 for j in range(self.output_dim[1])]
                bmu_idx = np.argmin(weights_dists)
                bmu_coords = np.array(np.unravel_index(bmu_idx, self.output_dim))

                for i in range(self.output_dim[0]):
                    for j in range(self.output_dim[1]):
                        delta = \
                            neighbourhood_func.val(bmu_coords, np.array([i, j]), epoch) * \
                            lr_decay_func(init_lr, epoch, epochs) * \
                            (x - self.weights[i][j])
                        self.weights[i][j] += delta
                        # for epoch statistics
                        if verbosity_period > 0:
                            deltas_abs.append(np.abs(delta))

            if verbosity_period > 0 and epoch % verbosity_period == 0:
                text = f'Epoch {epoch} done!\n' \
                       f'   - mean abs delta = {np.mean(deltas_abs)}\n' \
                       f'   - max abs delta = {np.max(deltas_abs)}\n' \
                       '============================================'
                print(text)

    def predict(self, data: np.ndarray) -> np.ndarray:
        activations = np.dot(self.weights, data.T)
        # flatten activations
        activations = activations.reshape(-1, activations.shape[-1])

        # find cluster labels for each input observation
        labels = np.argmax(activations, axis=0)
        return labels
