from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Optimiser(ABC):

    @abstractmethod
    def get_new_weights(self,
                        curr_weights: list[NDArray],
                        curr_biases: list[NDArray],
                        learning_rate: float,
                        batch_size: int,
                        weights_delta: list[NDArray],
                        biases_delta: list[NDArray]) \
            -> tuple[list[NDArray], list[NDArray]]:
        """
        :param curr_weights:  current weights
        :param curr_biases:   current biases
        :param learning_rate: a step size when updating weights
        :param batch_size:    a batch size used for training (how many times to perform
                              backpropagations before updating weights)
        :param weights_delta: weights delta obtained from backpropagation
        :param biases_delta:  biases delta obtained from backpropagation
        :return: A two-element tuple with the first element being the new weights
                 and the second being the new biases
        """
        pass


class BasicGD(Optimiser):
    """Basic gradient descent"""

    def get_new_weights(self,
                        curr_weights: list[NDArray],
                        curr_biases: list[NDArray],
                        learning_rate: float,
                        batch_size: int,
                        weights_delta: list[NDArray],
                        biases_delta: list[NDArray]) \
            -> tuple[list[NDArray], list[NDArray]]:

        new_weights = [w - learning_rate * delta / batch_size
                       for w, delta in zip(curr_weights, weights_delta)]
        new_biases = [b - learning_rate * delta / batch_size
                      for b, delta in zip(curr_biases, biases_delta)]

        return new_weights, new_biases


class MomentumGD(Optimiser):
    """Gradient descent with momentum"""

    __slots__ = ['lambda_coef', 'momentum_weights', 'momentum_biases']

    def __init__(self, lambda_coef: float = 0.9):
        self.lambda_coef = lambda_coef
        self.momentum_weights = []
        self.momentum_biases = []

    def get_new_weights(self,
                        curr_weights: list[NDArray],
                        curr_biases: list[NDArray],
                        learning_rate: float,
                        batch_size: int,
                        weights_delta: list[NDArray],
                        biases_delta: list[NDArray]) \
            -> tuple[list[NDArray], list[NDArray]]:

        if len(self.momentum_weights) == 0:
            self.momentum_weights = [np.zeros(w.shape) for w in curr_weights]
            self.momentum_biases = [np.zeros(b.shape) for b in curr_biases]

        # updating momentum
        self.momentum_weights = [w - self.lambda_coef * momentum
                                 for w, momentum in zip(weights_delta, self.momentum_weights)]
        self.momentum_biases = [b - self.lambda_coef * momentum
                                for b, momentum in zip(biases_delta, self.momentum_biases)]
        # updating weights
        new_weights = [w - learning_rate * momentum / batch_size
                       for w, momentum in zip(curr_weights, self.momentum_weights)]
        new_biases = [b - learning_rate * momentum / batch_size
                      for b, momentum in zip(curr_biases, self.momentum_biases)]

        return new_weights, new_biases


class RMSProp(Optimiser):
    """Root Mean Square Propagation"""

    __slots__ = ['beta_coef', 'eg2_weights', 'eg2_biases']

    def __init__(self, beta_coef: float = 0.9):
        self.beta_coef = beta_coef
        self.eg2_weights = []
        self.eg2_biases = []

    def get_new_weights(self,
                        curr_weights: list[NDArray],
                        curr_biases: list[NDArray],
                        learning_rate: float,
                        batch_size: int,
                        weights_delta: list[NDArray],
                        biases_delta: list[NDArray]) \
            -> tuple[list[NDArray], list[NDArray]]:

        if len(self.eg2_weights) == 0:
            self.eg2_weights = [np.zeros(w.shape) for w in curr_weights]
            self.eg2_biases = [np.zeros(b.shape) for b in curr_biases]

        eps = 10 ** (-100)

        # updating E[g^2]
        eg2_weights = [self.beta_coef * eg2 + (1 - self.beta_coef) * g ** 2
                       for eg2, g in zip(self.eg2_weights, weights_delta)]
        eg2_biases = [self.beta_coef * eg2 + (1 - self.beta_coef) * g ** 2
                      for eg2, g in zip(self.eg2_biases, biases_delta)]

        # updating weights
        new_weights = [w - learning_rate * (g / (np.sqrt(eg2) + eps)) / batch_size
                       for w, g, eg2 in zip(curr_weights, weights_delta, eg2_weights)]
        new_biases = [b - learning_rate * (g / (np.sqrt(eg2) + eps)) / batch_size
                      for b, g, eg2 in zip(curr_biases, biases_delta, eg2_biases)]

        return new_weights, new_biases
