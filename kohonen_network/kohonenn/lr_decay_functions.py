import numpy as np


def alpha_func(init_lr: float, epoch_no: int, epochs_total: int) -> float:
    """
    :param init_lr:         initial learning rate
    :param epoch_no:        current epoch number
    :param epochs_total:    total number of epochs in training
    :return:
    """
    return init_lr * np.exp(-epoch_no / epochs_total)
