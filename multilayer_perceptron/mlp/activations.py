from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

class ActivationFunction(ABC):

    @abstractmethod
    def val(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def deriv(self, x: NDArray) -> NDArray:
        pass


class SigmoidActivation(ActivationFunction):

    def val(self, x: NDArray) -> NDArray:
        return 1 / (np.exp(-x) + 1)

    def deriv(self, x: NDArray) -> NDArray:
        return self.val(x) * (1 - self.val(x))


class TanhActivation(ActivationFunction):

    def val(self, x: NDArray) -> NDArray:
        exp2x = np.exp(2 * x)
        return (exp2x - 1) / (exp2x + 1)

    def deriv(self, x: NDArray) -> NDArray:
        return 2 * np.exp(x) / (np.exp(2 * x) + 1)


class ReLUActivation(ActivationFunction):

    def val(self, x: NDArray) -> NDArray:
        return np.maximum(0, x)

    def deriv(self, x: NDArray) -> NDArray:
        return (x > 0) * 1


class LinearActivation(ActivationFunction):

    def val(self, x: NDArray) -> NDArray:
        return x

    def deriv(self, x: NDArray) -> NDArray:
        return np.ones(x.shape)


class SoftmaxActivation(ActivationFunction):

    def val(self, x: NDArray) -> NDArray:
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1).reshape(-1, 1)

    def deriv(self, x: NDArray) -> NDArray:
        return np.ones(x.shape)  # it works with cross-entropy

