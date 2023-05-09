from typing import Callable

import numpy as np
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .neighbourhood_func import NeighbourhoodFunction
from .distance import distance


class KohonenNetwork:
    __slots__ = ['input_size', 'output_dim', 'weights', 'hex_topology', '__rng',
                 '__normalise_min', '__normalise_max']

    def __init__(self, input_size: int, output_dim: tuple[int, int],
                 hex_topology: bool = False):
        self.input_size = input_size
        self.output_dim = output_dim
        self.hex_topology = hex_topology
        self.__reset_weights()

    @property
    def neurons_count(self) -> int:
        return self.output_dim[0] * self.output_dim[1]

    def __reset_weights(self, low: float = 0, high: float = 1, random_state: int = None):
        if random_state is None:
            rng = Generator(PCG64())
        else:
            rng = Generator(PCG64(random_state))
        self.__rng = rng

        self.weights = self.__rng.uniform(
            low=low, high=high,
            size=(self.neurons_count, self.input_size)
        )

    def __shuffled_vector(self, v: np.ndarray) -> np.ndarray:
        """Returns a shuffled copy of a given vector"""
        indices = np.arange(v.shape[0])
        self.__rng.shuffle(indices)
        return v[indices]

    def visualise_centroids(self, data: np.ndarray, fig: plt.Figure = None):
        if fig is None:
            fig = plt.figure()

        # handle multi-dimensional input data
        if self.input_size == 2:
            data_x = data[:, 0].reshape(1, -1)
            data_y = data[:, 1].reshape(1, -1)
            weights_x = self.weights[:, 0].reshape(1, -1)
            weights_y = self.weights[:, 1].reshape(1, -1)
        else:
            dim_reducer = TSNE(n_components=2)
            data_and_weights = np.concatenate([self.weights, data])
            data_and_weights_2d = dim_reducer.fit_transform(data_and_weights)

            data_x = data_and_weights_2d[self.neurons_count:, 0].reshape(1, -1)
            data_y = data_and_weights_2d[self.neurons_count:, 1].reshape(1, -1)
            weights_x = data_and_weights_2d[:self.neurons_count, 0].reshape(1, -1)
            weights_y = data_and_weights_2d[:self.neurons_count, 1].reshape(1, -1)

        # plotting data
        plt.scatter(data_x, data_y, c=self.predict(data))

        # centroid marker properties
        marker_type = 'x'
        size_outer = 140
        size_inner = 100
        colour_outer = 'black'
        colour_inner = 'white'
        width_outer = 4
        width_inner = 1.5

        # plotting centroids
        plt.scatter(
            weights_x, weights_y, 
            s=size_outer, 
            marker=marker_type, 
            color=colour_outer, 
            linewidths=width_outer,
        )
        plt.scatter(
            weights_x, weights_y, 
            s=size_inner, 
            marker=marker_type, 
            color=colour_inner, 
            linewidths=width_inner,
        )

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
        self.__reset_weights(np.min(data) * 1.1, np.max(data) * 1.1, random_state)

        for epoch in range(epochs):
            # epochs start from 1
            epoch += 1
            deltas_abs = []

            for x in self.__shuffled_vector(data):
                weights_dists = [distance(self.weights[i], x)
                                 for i in range(self.neurons_count)]
                bmu_idx = np.argmin(weights_dists)
                bmu_coords = np.array(np.unravel_index(bmu_idx, self.output_dim))

                for i in range(self.neurons_count):
                    # convert i to topological coordinates
                    curr_coords = np.array(np.unravel_index(i, self.output_dim))
                    # calculate distance to bmu
                    bmu_dist = distance(
                        bmu_coords, curr_coords,
                        a_to_hex=self.hex_topology,
                        b_to_hex=self.hex_topology,
                    )
                    # calculate weights delta
                    delta = \
                        neighbourhood_func.val(bmu_dist, epoch) * \
                        lr_decay_func(init_lr, epoch, epochs) * \
                        (x - self.weights[i])
                    self.weights[i] += delta
                    # for epoch statistics
                    if verbosity_period > 0 and epoch % verbosity_period == 0:
                        deltas_abs.append(np.abs(delta))

            if verbosity_period > 0 and epoch % verbosity_period == 0:
                text = f'Epoch {epoch} done!\n' \
                       f'   - mean abs delta = {np.mean(deltas_abs)}\n' \
                       f'   - max abs delta = {np.max(deltas_abs)}\n' \
                       '============================================'
                print(text)

    def predict(self, data: np.ndarray) -> np.ndarray:
        labels = []
        for x in data:
            weights_dists = [distance(self.weights[i], x)
                             for i in range(self.neurons_count)]
            bmu_idx = np.argmin(weights_dists)
            labels.append(bmu_idx)
        return np.array(labels)
