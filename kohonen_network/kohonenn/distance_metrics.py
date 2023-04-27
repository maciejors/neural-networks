import numpy as np


def euclidean_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y)
