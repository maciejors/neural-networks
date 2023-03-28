from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as nptypes


class LossFunction(ABC):

    @abstractmethod
    def val(self, y_true: nptypes.NDArray, y_pred: nptypes.NDArray) -> float:
        pass


class MSE(LossFunction):

    def val(self, y_true: nptypes.NDArray, y_pred: nptypes.NDArray) -> float:
        n = len(y_true)
        return (1 / n) * np.sum((y_true - y_pred) ** 2)


class CrossEntropy(LossFunction):
    eps = 10 ** (-100)

    def val(self, y_true: nptypes.NDArray, y_pred: nptypes.NDArray) -> float:
        return -np.sum(y_true * np.log(y_pred + self.eps))
