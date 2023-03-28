from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptypes


class ActivationFunction(ABC):

    @abstractmethod
    def val(self, x: nptypes.NDArray) -> nptypes.NDArray:
        pass

    @abstractmethod
    def deriv(self, x: nptypes.NDArray) -> nptypes.NDArray:
        pass


class SigmoidActivation(ActivationFunction):

    def val(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return 1 / (np.exp(-x) + 1)

    def deriv(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return self.val(x) * (1 - self.val(x))


class LinearActivation(ActivationFunction):

    def val(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return x

    def deriv(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return np.ones(x.shape)
