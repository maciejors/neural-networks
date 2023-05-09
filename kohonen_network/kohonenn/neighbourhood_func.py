from abc import ABC, abstractmethod

import numpy as np


class NeighbourhoodFunction(ABC):
    __slots__ = ['nbhood_width_factor']

    def __init__(self, nbhood_width_factor: int = 1):
        self.nbhood_width_factor = nbhood_width_factor

    @abstractmethod
    def val(self, d: float, t: int) -> float:
        pass


class GaussFunction(NeighbourhoodFunction):

    def val(self, d: float, t: int) -> float:
        d_scaled = d * self.nbhood_width_factor
        return np.exp(-(d_scaled * t) ** 2)


class MexicanHat(NeighbourhoodFunction):

    def val(self, d: float, t: int) -> float:
        d_scaled = d * self.nbhood_width_factor
        return (2 - 4 * d_scaled ** 2) * np.exp(-(d_scaled * t) ** 2)
