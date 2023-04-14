from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import seaborn as sns

from .activations import ActivationFunction, ReLUActivation, SigmoidActivation, TanhActivation
from .lossfunc_and_metrics import LossFunction
from .optimisers import Optimiser, MomentumGD


def _get_rng(seed: int | str | None) -> Generator:
    if seed is not None:
        if type(seed) is str:
            seed = np.sum([ord(c) for c in seed])
        rng = Generator(PCG64(seed))
    else:
        rng = Generator(PCG64())
    return rng


def _generate_batches(x: NDArray, y: NDArray, batch_size: int,
                      rng: Generator) -> tuple[list[NDArray], list[NDArray]]:
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


class MLP:
    __slots__ = ['input_size', 'output_size', 'hidden_layers_sizes', 'weights', 'biases',
                 'activation_func', 'out_func', 'loss', 
                 '__normalise_max_x', '__normalise_min_x', '__normalise_max_y', '__normalise_min_y']

    def __init__(self, input_size: int, hidden_layers_sizes: list[int], output_size: int,
                 activation_func: ActivationFunction, out_func: ActivationFunction,
                 loss: LossFunction, weights_init_method: Literal['default', 'xavier', 'he'] = None):
        """
        :param input_size:          dimension of the input
        :param hidden_layers_sizes: sizes of the hidden layers passed as a list (e.g. [10, 10])
        :param output_size:         dimension of the output
        :param activation_func:     activation function for all the layers except the last one
        :param out_func:            activation function for the last layer
        :param loss:                a loss function optimised during training
        :param weights_init_method: a method to use to initialise weights. Possible values:
                                    "default", "xavier", "he". Default means the weights will be
                                    initialised from the uniform distribution on a [0, 1] interval.
                                    If set to None, the method will be inferred from the used
                                    activation function ("he" for ReLU, "xavier" for sigmoid or
                                    tanh, "default" otherwise)
        """
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.activation_func = activation_func
        self.out_func = out_func
        self.loss = loss

        self.weights = []
        self.biases = []
        self.__set_default_weights(method=weights_init_method)

        # default normalisation values mean 'no normalisation':
        self.__normalise_min_x = 0
        self.__normalise_max_x = 1
        self.__normalise_min_y = 0
        self.__normalise_max_y = 1

    def __set_default_weights(self, method: str = None, rng: Generator = None):
        self.weights = []
        self.biases = []

        if rng is None:
            rng = Generator(PCG64())
        if method is None:
            if type(self.activation_func) == ReLUActivation:
                method = 'he'
            elif type(self.activation_func) == SigmoidActivation or \
                    type(self.activation_func) == TanhActivation:
                method = 'xavier'
            else:
                method = 'default'

        layers_sizes = [self.input_size, *self.hidden_layers_sizes, self.output_size]
        for i in range(len(layers_sizes) - 1):
            shape = (layers_sizes[i], layers_sizes[i + 1])

            if method == 'xavier':
                xavier_boundary = 6 / np.sqrt(shape[0] + shape[1])
                new_weights = rng.uniform(low=-xavier_boundary, high=xavier_boundary, size=shape)
            elif method == 'he':
                new_weights = rng.normal(scale=np.sqrt(2 / shape[0]), size=shape)
            else:
                # uniform [0, 1] distribution by default
                new_weights = rng.uniform(size=shape)

            self.weights.append(new_weights)

        # default biases are 0 by default
        for layer_size in layers_sizes[1:]:
            self.biases.append(
                np.zeros((1, layer_size))
            )

    @property
    def layers_count(self) -> int:
        return len(self.weights) + 1

    def visualise(self, nrow: int = None, ncol: int = None, annotate: bool = True):
        """
        Visualises weights and biases using heatmaps (weights and biases from a single layer
        are concatenated to make the visualisation more condensed)

        :param nrow:     number of heatmaps per column
        :param ncol:     number of heatmaps per row
        :param annotate: whether to add numerical values of weights and biases on top of every box
                         on a heatmap
        """
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

    def __normalise_x(self, x: NDArray) -> NDArray:
        return (x - self.__normalise_min_x) / (self.__normalise_max_x - self.__normalise_min_x)

    def __normalise_y(self, y: NDArray) -> NDArray:
        return (y - self.__normalise_min_y) / (self.__normalise_max_y - self.__normalise_min_y)

    def __denormalise_x(self, x: NDArray) -> NDArray:
        return x * (self.__normalise_max_x - self.__normalise_min_x) + self.__normalise_min_x

    def __denormalise_y(self, y: NDArray) -> NDArray:
        return y * (self.__normalise_max_y - self.__normalise_min_y) + self.__normalise_min_y

    def __handle_verbosity(self, verbosity_period: int, epoch: int, total_epochs: int,
                           x_train_pre_norm: NDArray, y_train_pre_norm: NDArray,
                           x_test: NDArray, y_test: NDArray):
        # printing the progress every [verbosity_period]th epoch
        # unless verbosity_period is set to 0
        if verbosity_period > 0 and (epoch + 1) % verbosity_period == 0:

            loss_train = self.loss.val(y_train_pre_norm, self.predict(x_train_pre_norm))
            text = f'Epoch {epoch + 1}/{total_epochs} done | ' \
                   f'loss(train) = {loss_train:.3f}'

            if x_test is not None and y_test is not None:
                loss_test = self.loss.val(y_test, self.predict(x_test))
                text += f' | loss(test) = {loss_test:.3f}'
            print(text)

    def __feedforward(self, x: NDArray) -> tuple[list[NDArray], list[NDArray]]:
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

    def __backpropagate(self, y: NDArray, z: list[NDArray], a: list[NDArray]) \
            -> tuple[list[NDArray], list[NDArray]]:
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

    def train(self, x_train: NDArray, y_train: NDArray, epochs: int,
              x_test: NDArray = None, y_test: NDArray = None,
              batch_size: int = 32,
              learning_rate: float | Callable = 0.001,
              optimiser: Optimiser = None,
              weights_init_method: str = None,
              plot_loss: bool = False,
              random_state: int | None = None,
              verbosity_period: int = 0,
              l2_coef: float = 0,
              stop_on_test_loss_rise = False) -> tuple[list[float], list[float]]:
        """
        Attempts to find optimal weights & biases for the network

        :param x_train:                data used for training
        :param y_train:                target variable values used for training
        :param epochs:                 maximum number of epoch after which the training process
                                       will stop
        :param x_test:                 data used for test evalutaion
        :param y_test:                 target variable values used for test evalutaion
        :param batch_size:             a batch size used for training (how many times to perform
                                       backpropagations before updating weights)
        :param optimiser:              an optimiser used to update weights
        :param weights_init_method:    a method to use to initialise weights. Possible values:
                                       "default", "xavier", "he". If set to None, the method
                                       will be inferred from the used activation function ("he" for
                                       ReLU, "xavier" for sigmoid or tanh, "default" otherwise)
        :param learning_rate:          a learning rate coefficient. Can be either a number or a
                                       function that takes the epoch number as an argument and
                                       returns a float
        :param random_state:           a seed passed to a random number generation (affects the
                                       randomness of generating batches and setting initial
                                       weights & biases)
        :param plot_loss:              whether to plot loss function values per epoch after
                                       the training
        :param verbosity_period:       controls how often loss values are printed. If
                                       verbosity_period = n, then the progress will be printed
                                       every nth epoch. If verbosity_period = 0, then nothing
                                       will be printed
        :param l2_coef:                lambda coefficient for L2 regularisation. Set to 0 for
                                       no regularisation
        :param stop_on_test_loss_rise: whether to stop the training process when the loss value on a 
                                       test dataset gets doubled compared to the previous epoch

        :return:                       a tuple containing two lists with loss values after every epoch,
                                       the first one contains loss values on a train data and the
                                       second one contains loss values on the test data. The second
                                       list will be empty if no test data has been provided.
        """
        # default optimiser
        if optimiser is None:
            optimiser = MomentumGD()

        # random_state for reproducibility
        rng = _get_rng(random_state)

        # reset weights
        self.__set_default_weights(rng=rng, method=weights_init_method)

        # normalisation
        self.__normalise_max_x = np.max(x_train)
        self.__normalise_min_x = np.min(x_train)
        self.__normalise_max_y = np.max(y_train)
        self.__normalise_min_y = np.min(y_train)
        x_norm = self.__normalise_x(x_train)
        y_norm = self.__normalise_y(y_train)

        # to avoid code repetition
        is_test_data_provided = x_test is not None and y_test is not None

        # train loss after every epoch
        loss_history_train = [self.loss.val(y_train, self.predict(x_train))]
        loss_history_test = []
        if is_test_data_provided:
            loss_history_test.append(self.loss.val(y_test, self.predict(x_test)))

        # adjusting weights & biases using backpropagation
        for epoch in range(epochs):
            # save weights from the start of the epoch
            weights_epoch_start = self.weights
            biases_epoch_start = self.biases

            # set the learning rate
            if type(learning_rate) is float or type(learning_rate) is int:
                lr = learning_rate
            else:
                lr = learning_rate(epoch)

            # generate batches
            x_batches, y_batches = _generate_batches(x_norm, y_norm, batch_size, rng)

            # iterate over batches
            for x, y in zip(x_batches, y_batches):
                # FeedForward
                # z - values before activation
                # a - values after activation
                z, a = self.__feedforward(x)

                # Backpropagate
                weights_delta, biases_delta = self.__backpropagate(y, z, a)

                # Regularisation
                if l2_coef != 0:
                    weights_delta = [l2_coef * old_w + new_w for old_w, new_w in zip(self.weights, weights_delta)]

                # updating weights
                self.weights, self.biases = optimiser.get_new_weights(
                    curr_weights=self.weights, curr_biases=self.biases,
                    learning_rate=lr, batch_size=len(y),
                    weights_delta=weights_delta, biases_delta=biases_delta
                )
            
            # stop the training if test loss value gets too high
            if stop_on_test_loss_rise and is_test_data_provided:
                curr_test_loss = self.loss.val(y_test, self.predict(x_test))
                prev_test_loss = loss_history_test[-1]
                if curr_test_loss > 2 * prev_test_loss:
                    self.weights = weights_epoch_start
                    self.biases = biases_epoch_start
                    print(f'Test loss too high (new loss: {curr_test_loss:.2f}; old loss: {prev_test_loss:.2f}) - stopping the training')
                    break

            # saving loss
            loss_history_train.append(self.loss.val(y_train, self.predict(x_train)))
            if is_test_data_provided:
                loss_history_test.append(self.loss.val(y_test, self.predict(x_test)))

            # printing the progress
            self.__handle_verbosity(verbosity_period=verbosity_period, epoch=epoch,
                                    total_epochs=epochs,
                                    x_train_pre_norm=x_train, y_train_pre_norm=y_train,
                                    x_test=x_test, y_test=y_test)
        # summary
        min_loss_epoch = np.argmin(loss_history_train)
        min_loss = loss_history_train[min_loss_epoch]
        print(f'Min train loss: {min_loss:.3f} (epoch {min_loss_epoch})')
        print(f'Final train metric value: {loss_history_train[-1]:.3f}')

        if plot_loss:               
            plt.plot(list(range(len(loss_history_train))), loss_history_train)
            plt.yscale('log')
            plt.ylabel('Train loss value')
            plt.xlabel('Epoch')
            plt.show()
        return loss_history_train, loss_history_test

    def predict(self, x: NDArray, convert_prob_to_labels: bool = False) -> NDArray:
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
