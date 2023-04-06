from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score


class LossFunction(ABC):

    @abstractmethod
    def val(self, y_true: NDArray, y_pred: NDArray) -> float:
        pass


class MSE(LossFunction):

    def val(self, y_true: NDArray, y_pred: NDArray) -> float:
        n = len(y_true)
        return (1 / n) * np.sum((y_true - y_pred) ** 2)


class CrossEntropy(LossFunction):
    eps = 10 ** (-100)

    def val(self, y_true: NDArray, y_pred: NDArray) -> float:
        return -np.sum(y_true * np.log(y_pred + self.eps))


def fmeasure_prob(true_prob: NDArray, pred_prob: NDArray) -> float:
    """
    F1 Score for probability predictions

    :param true_prop:              true probabilities
    :param pred_prob:              predicted probabilities
    """
    true_labels = np.argmax(true_prob, axis=1).reshape(-1, 1)
    pred_labels = np.argmax(pred_prob, axis=1).reshape(-1, 1)
    return f1_score(true_labels, pred_labels, average='micro')
