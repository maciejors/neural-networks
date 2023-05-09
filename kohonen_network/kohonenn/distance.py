import numpy as np


def distance(a: np.ndarray, b: np.ndarray,
             a_to_hex=False, b_to_hex=False) -> float:
    """
    :param a:           a two-element array
    :param b:           a two-element array
    :param a_to_hex:    set to true if `a` represent weights and a hexagonal topology is used
    :param b_to_hex:    set to true if `b` represent weights and a hexagonal topology is used
    :return: a distance from `a` to `b`
    """
    if a_to_hex:
        a = __convert_to_hex(a)
    if b_to_hex:
        b = __convert_to_hex(b)
    return np.linalg.norm(a - b)


def __convert_to_hex(idx_coord: np.ndarray) -> np.ndarray:
    """Adjusts single point coordinates to the hexagonal topology"""
    if idx_coord[1] % 2 == 0:
        hex_x = idx_coord[0] * np.sqrt(3)
    else:
        hex_x = idx_coord[0] * np.sqrt(3) + np.sqrt(3) / 2
    hex_y = idx_coord[1] * 3/2

    return np.array([hex_x, hex_y])
