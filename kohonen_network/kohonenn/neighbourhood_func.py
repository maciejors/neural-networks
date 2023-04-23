from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .distance_metrics import euclidean_dist


class NeighbourhoodFunction(ABC):
    __slots__ = ['distance_func', 'nbhood_width_factor']

    def __init__(self, nbhood_width_factor: int = 1,
                 distance_metric: Callable = None):
        if distance_metric is None:
            distance_metric = euclidean_dist
        self.distance_func = distance_metric
        self.nbhood_width_factor = nbhood_width_factor

    @abstractmethod
    def val(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        pass


class GaussFunction(NeighbourhoodFunction):

    def val(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        d = self.nbhood_width_factor * self.distance_func(x, y)
        return np.exp(-(d * t) ** 2)


class MexicanHat(NeighbourhoodFunction):

    def val(self, x: np.ndarray, y: np.ndarray, t: int) -> np.ndarray:
        d = self.nbhood_width_factor * self.distance_func(x, y)
        # return (2 - 4 * t**2 * d**2) * t**2 * np.exp(-t**2 * d**2)
        return (2 - 4 * d**2) * np.exp(-d**2)
