from typing import Callable

import numpy as np
import numpy.typing as nptypes
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import seaborn as sns

from .activations import ActivationFunction
from .lossfunc import LossFunction


class MLP:
    __slots__ = ['input_size', 'output_size', 'hidden_layers_sizes', 'weights', 'biases',
                 'activation_func', 'out_func',
                 'loss', '__normalise_max_x', '__normalise_min_x', '__normalise_max_y',
                 '__normalise_min_y']

    def __init__(self, input_size: int, hidden_layers_sizes: list[int], output_size: int,
                 activation_func: ActivationFunction, out_func: ActivationFunction,
                 loss: LossFunction):
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.activation_func = activation_func
        self.out_func = out_func
        self.loss = loss
        self.__set_default_weights()

    def __set_default_weights(self, rng: Generator = None):
        self.weights = []
        self.biases = []

        if rng is None:
            rng = Generator(PCG64())

        # default weigths are taken from normal distribution
        layers_sizes = [self.input_size, *self.hidden_layers_sizes, self.output_size]
        for i in range(len(layers_sizes) - 1):
            self.weights.append(
                rng.uniform(size=(layers_sizes[i], layers_sizes[i + 1]))
            )
        # default biases are 0
        for layer_size in layers_sizes[1:]:
            self.biases.append(
                np.zeros((1, layer_size))
            )

    @property
    def layers_count(self) -> int:
        return len(self.weights) + 1

    def visualise(self, nrow: int = None, ncol: int = None, annotate: bool = True):
        if ncol is None and nrow is None:
            nrow = 1
            ncol = self.layers_count - 1
        elif ncol is None:
            ncol = int(np.ceil((self.layers_count - 1) / nrow))
        elif nrow is None:
            nrow = int(np.ceil((self.layers_count - 1) / ncol))

        for i, weights, biases in zip(range(self.layers_count - 1), self.weights, self.biases):
            # concatenating weights & biases
            weights = weights.reshape(len(biases[0]), -1)
            biases = biases.reshape(-1, 1)
            layer = np.concatenate([weights, biases], axis=1)

            # plotting layer values on a heatmap
            plt.subplot(nrow, ncol, i + 1)
            sns.heatmap(layer, cmap='BrBG', annot=annotate)
            plot_title = f'{i} -> {i + 1}'
            if i == 0:
                plot_title = 'Input -> 1'
            elif i == self.layers_count - 2:
                plot_title = f'{i} -> Output'
            plt.title(plot_title)
        plt.show()

    def __normalise_x(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return (x - self.__normalise_min_x) / (self.__normalise_max_x - self.__normalise_min_x)

    def __normalise_y(self, y: nptypes.NDArray) -> nptypes.NDArray:
        return (y - self.__normalise_min_y) / (self.__normalise_max_y - self.__normalise_min_y)

    def __denormalise_x(self, x: nptypes.NDArray) -> nptypes.NDArray:
        return x * (self.__normalise_max_x - self.__normalise_min_x) + self.__normalise_min_x

    def __denormalise_y(self, y: nptypes.NDArray) -> nptypes.NDArray:
        return y * (self.__normalise_max_y - self.__normalise_min_y) + self.__normalise_min_y

    def __get_rng(self, seed: int | str | None) -> Generator:
        if seed is not None:
            if type(seed) is str:
                seed = np.sum([ord(c) for c in seed])
            rng = Generator(PCG64(seed))
        else:
            rng = Generator(PCG64())
        return rng

    def __generate_batches(self, x: nptypes.NDArray, y: nptypes.NDArray, batch_size: int,
                           rng: Generator) -> list[tuple[nptypes.NDArray, nptypes.NDArray]]:
        data_size = y.shape[0]
        # shuffle x & y
        indices = np.arange(data_size)
        rng.shuffle(indices)
        x = x[indices]
        y = y[indices]
        # generate batches
        split_points = np.arange(batch_size, data_size, batch_size)
        x_batches = np.array_split(x, split_points)
        y_batches = np.array_split(y, split_points)
        return x_batches, y_batches

    def __handle_verbosity(self, verbosity_period: int, epoch: int, total_epochs: int,
                           x_train_pre_norm: nptypes.NDArray, y_train_pre_norm: nptypes.NDArray,
                           x_test: nptypes.NDArray, y_test: nptypes.NDArray):
        # printing the progress every [verbosity_period]th epoch, unless verbosity_period is set to 0
        if verbosity_period > 0 and (epoch + 1) % verbosity_period == 0:

            loss_train = self.loss.val(y_train_pre_norm, self.predict(x_train_pre_norm))
            text = f'Epoch {epoch + 1}/{total_epochs} done | loss(train) = {loss_train:.2f}'

            if x_test is not None and y_test is not None:
                loss_test = self.loss.val(y_test, self.predict(x_test))
                text += f' | loss(test) = {loss_test:.2f}'
            print(text)

    def __feedforward(self, x: nptypes.NDArray) -> tuple[
        list[nptypes.NDArray], list[nptypes.NDArray]]:
        """
        Computes and returns the values in every layer from before and after activation

        :return: A two-element tuple, where the first element is a list of values before activation,
                 and the second is a list of values after activation
        """
        prev_a = x
        z = []
        a = [x]

        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            z_k = (prev_a @ weights) + biases
            z.append(z_k)
            prev_a = self.activation_func.val(z_k)
            a.append(prev_a)
        # the last iteration uses a different activation function
        z_k = (prev_a @ self.weights[-1]) + self.biases[-1]
        z.append(z_k)
        prev_a = self.out_func.val(z_k)
        a.append(prev_a)
        return z, a

    def __backpropagate(self, y: nptypes.NDArray,
                        z: list[nptypes.NDArray],
                        a: list[nptypes.NDArray]) -> tuple[
        list[nptypes.NDArray], list[nptypes.NDArray]]:
        """
        Performs backpropagation and returns weights and biases deltas

        :param z: values before activation
        :param a: values after activation
        :return: A tuple, where the first element is the weights_delta and the second one is the biases_delta
        """
        # initialise deltas to 0
        weights_delta = [np.zeros(w.shape) for w in self.weights]
        biases_delta = [np.zeros(b.shape) for b in self.biases]

        y_pred = a[-1]
        e_k = (y_pred - y) * self.out_func.deriv(z[-1])
        weights_delta[-1] = a[-2].T @ e_k
        biases_delta[-1] = e_k.sum(axis=0)

        for k in range(2, self.layers_count):
            e_k = self.activation_func.deriv(z[-k]) * (self.weights[-k + 1] @ e_k.T).T
            weights_delta[-k] = a[-k - 1].T @ e_k
            biases_delta[-k] = e_k.sum(axis=0)

        return weights_delta, biases_delta

    def __summarise_training(self, loss_history: list[float], plot_loss: bool):
        min_loss = np.min(loss_history)
        min_loss_epoch = loss_history.index(min_loss)
        print(f'Minimal train loss: {min_loss:.2f} (epoch {min_loss_epoch})')
        print(f'Final train loss: {loss_history[-1]:.2f}')

        if plot_loss:
            plt.figure(figsize=(10, 8))
            plt.plot(list(range(len(loss_history))), loss_history)
            plt.yscale('log')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

    def __gradient_descent_basic(self, x_norm: nptypes.NDArray, y_norm: nptypes.NDArray,
                                 x_test: nptypes.NDArray, y_test: nptypes.NDArray,
                                 epochs: int, learning_rate: float | Callable, batch_size: int,
                                 verbosity_period: int, rng: Generator) -> list[float]:
        """
        :return: loss values on the training dataset after every epoch
        """
        x_pre_norm = self.__denormalise_x(x_norm)
        y_pre_norm = self.__denormalise_y(y_norm)

        # train loss after every epoch
        loss_history = [self.loss.val(y_pre_norm, self.predict(x_pre_norm))]

        # adjusting weights & biases using backpropagation
        for epoch in range(epochs):
            # set the learning rate
            if type(learning_rate) is float or type(learning_rate) is int:
                lr = learning_rate
            else:
                lr = learning_rate(epoch)

            # generate batches
            x_batches, y_batches = self.__generate_batches(x_norm, y_norm, batch_size, rng)

            # iterate over batches
            for x, y in zip(x_batches, y_batches):
                # FeedForward
                # z - values before activation
                # a - values after activation
                z, a = self.__feedforward(x)

                # Backpropagate
                weights_delta, biases_delta = self.__backpropagate(y, z, a)

                # updating weights
                self.weights = [w - lr * delta / len(y)
                                for w, delta in zip(self.weights, weights_delta)]
                self.biases = [b - lr * delta / len(y)
                               for b, delta in zip(self.biases, biases_delta)]

            # saving loss
            loss_history.append(self.loss.val(y_pre_norm, self.predict(x_pre_norm)))

            # printing the progress
            self.__handle_verbosity(verbosity_period=verbosity_period, epoch=epoch,
                                    total_epochs=epochs,
                                    x_train_pre_norm=x_pre_norm, y_train_pre_norm=y_pre_norm,
                                    x_test=x_test, y_test=y_test)
        return loss_history

    def __gradient_descent_momentum(self, x_norm: nptypes.NDArray, y_norm: nptypes.NDArray,
                                    x_test: nptypes.NDArray, y_test: nptypes.NDArray,
                                    epochs: int, learning_rate: float | Callable, batch_size: int,
                                    lambda_coef: float,
                                    verbosity_period: int, rng: Generator) -> list[float]:
        """
        :return: loss values on the training dataset after every epoch
        """
        x_pre_norm = self.__denormalise_x(x_norm)
        y_pre_norm = self.__denormalise_y(y_norm)

        # train loss after every epoch
        loss_history = [self.loss.val(y_pre_norm, self.predict(x_pre_norm))]

        # momentum
        momentum_weights = [np.zeros(w.shape) for w in self.weights]
        momentum_biases = [np.zeros(b.shape) for b in self.biases]

        # adjusting weights & biases using backpropagation
        for epoch in range(epochs):
            # set the learning rate
            if type(learning_rate) is float or type(learning_rate) is int:
                lr = learning_rate
            else:
                lr = learning_rate(epoch)

            # generate batches
            x_batches, y_batches = self.__generate_batches(x_norm, y_norm, batch_size, rng)

            # iterate over batches
            for x, y in zip(x_batches, y_batches):
                # FeedForward
                # z - values before activation
                # a - values after activation
                z, a = self.__feedforward(x)

                # Backpropagate
                weights_delta, biases_delta = self.__backpropagate(y, z, a)

                # updating momentum
                momentum_weights = [w - lambda_coef * momentum
                                    for w, momentum in zip(weights_delta, momentum_weights)]
                momentum_biases = [b - lambda_coef * momentum
                                   for b, momentum in zip(biases_delta, momentum_biases)]
                # updating weights
                self.weights = [w - lr * momentum / len(y)
                                for w, momentum in zip(self.weights, weights_delta)]
                self.biases = [b - lr * momentum / len(y)
                               for b, momentum in zip(self.biases, biases_delta)]

            # saving loss
            loss_history.append(self.loss.val(y_pre_norm, self.predict(x_pre_norm)))

            # printing the progress
            self.__handle_verbosity(verbosity_period=verbosity_period, epoch=epoch,
                                    total_epochs=epochs,
                                    x_train_pre_norm=x_pre_norm, y_train_pre_norm=y_pre_norm,
                                    x_test=x_test, y_test=y_test)
        return loss_history

    def __rmsprop(self, x_norm: nptypes.NDArray, y_norm: nptypes.NDArray,
                  x_test: nptypes.NDArray, y_test: nptypes.NDArray,
                  epochs: int, learning_rate: float | Callable, batch_size: int, beta_coef: float,
                  verbosity_period: int, rng: Generator) -> list[float]:
        """
        :return: loss values on the training dataset after every epoch
        """
        x_pre_norm = self.__denormalise_x(x_norm)
        y_pre_norm = self.__denormalise_y(y_norm)

        # train loss after every epoch
        loss_history = [self.loss.val(y_pre_norm, self.predict(x_pre_norm))]

        # E[g^2]
        eg2_weights = [np.zeros(w.shape) for w in self.weights]
        eg2_biases = [np.zeros(b.shape) for b in self.biases]

        # adjusting weights & biases using backpropagation
        for epoch in range(epochs):
            # set the learning rate
            if type(learning_rate) is float or type(learning_rate) is int:
                lr = learning_rate
            else:
                lr = learning_rate(epoch)

            # generate batches
            x_batches, y_batches = self.__generate_batches(x_norm, y_norm, batch_size, rng)

            # iterate over batches
            for x, y in zip(x_batches, y_batches):
                # FeedForward
                # z - values before activation
                # a - values after activation
                z, a = self.__feedforward(x)

                # Backpropagate
                g_weights, g_biases = self.__backpropagate(y, z, a)

                # updating E[g^2]
                eg2_weights = [beta_coef * eg2 + (1 - beta_coef) * g ** 2
                               for eg2, g in zip(eg2_weights, g_weights)]
                eg2_biases = [beta_coef * eg2 + (1 - beta_coef) * g ** 2
                              for eg2, g in zip(eg2_biases, g_biases)]

                # updating weights
                self.weights = [w - lr * (g / np.sqrt(eg2)) / len(y)
                                for w, g, eg2 in zip(self.weights, g_weights, eg2_weights)]
                self.biases = [b - lr * (g / np.sqrt(eg2)) / len(y)
                               for b, g, eg2 in zip(self.biases, g_biases, eg2_biases)]

            # saving loss
            loss_history.append(self.loss.val(y_pre_norm, self.predict(x_pre_norm)))

            # printing the progress
            self.__handle_verbosity(verbosity_period=verbosity_period, epoch=epoch,
                                    total_epochs=epochs,
                                    x_train_pre_norm=x_pre_norm, y_train_pre_norm=y_pre_norm,
                                    x_test=x_test, y_test=y_test)
        return loss_history

    def train(self, x_train: nptypes.NDArray, y_train: nptypes.NDArray,
              x_test: nptypes.NDArray = None, y_test: nptypes.NDArray = None, method='basic',
              epochs: int = 1000, learning_rate: float | Callable = 0.001,
              batch_size: int = 1, momentum_coef: float = 0.9, rmsprop_coef: float = 0.9,
              plot_loss: bool = False, random_state: int | None = None,
              verbosity_period: int = 0) -> list[float]:
        """
        Attempts to find optimal weights & biases for the network

        :param method:                 a method used to find the optimal weights. Accepted values: 'basic' (basic gradient descent),
                                       'momentum' (gradient descent with momentum), 'rmsprop'.
        :param learning_rate:          a learning rate coefficient. Can be either a number or a function that takes the epoch
                                       number as an argument and returns a float
        :param momentum_coef:          a lambda coefficient for the gradient descent with momentum algorithm. Ignored unless the method
                                       parameter is set to 'momentum'
        :param rmsprop_coef:           a beta coefficient for the RMSProp algorithm. Ignored unless the method parameter is set to 'rmsprop'
        :param random_state:           a seed passed to a random number generation (affects the randomness of generating batches and
                                       setting initial weights & biases)
        :param plot_loss:              whether to plot loss function values per epoch after the training
        :param verbosity_period:       controls how often loss values are printed. If verbosity_period = n, then the progress will be printed
                                       every nth epoch. If verbosity_period = 0, then nothing will be printed.

        :return:                       loss values after every epoch
        """
        # random_state for reproducibility
        rng = self.__get_rng(random_state)

        # reset weights
        self.__set_default_weights(rng)

        # normalisation
        self.__normalise_max_x = np.max(x_train)
        self.__normalise_min_x = np.min(x_train)
        self.__normalise_max_y = np.max(y_train)
        self.__normalise_min_y = np.min(y_train)
        x_norm = self.__normalise_x(x_train)
        y_norm = self.__normalise_y(y_train)

        # finding the best weights & biases
        if method == 'basic':
            loss_history = self.__gradient_descent_basic(
                x_norm=x_norm, y_norm=y_norm, x_test=x_test, y_test=y_test,
                epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size,
                verbosity_period=verbosity_period, rng=rng,
            )
        elif method == 'momentum':
            loss_history = self.__gradient_descent_momentum(
                x_norm=x_norm, y_norm=y_norm, x_test=x_test, y_test=y_test,
                epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size, lambda_coef=momentum_coef,
                verbosity_period=verbosity_period, rng=rng,
            )
        elif method == 'rmsprop':
            loss_history = self.__rmsprop(
                x_norm=x_norm, y_norm=y_norm, x_test=x_test, y_test=y_test,
                epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size, beta_coef=rmsprop_coef,
                verbosity_period=verbosity_period, rng=rng,
            )
        else:
            raise ValueError(
                'Wrong value for the method parameter. Possible values: "basic", "momentum", "rmsprop"')

        # summary
        self.__summarise_training(loss_history, plot_loss)
        return loss_history

    def predict(self, x: nptypes.NDArray,
                convert_prob_to_labels: bool = False) -> nptypes.NDArray:
        # normalisation
        x = self.__normalise_x(x)

        # prediction
        _, activations = self.__feedforward(x)
        y = activations[-1]

        # denormalisation
        y = self.__denormalise_y(y)

        if convert_prob_to_labels:
            y = np.argmax(y, axis=1).reshape(-1, 1)

        return y