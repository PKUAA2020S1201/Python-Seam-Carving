import numpy as np
import sc2


def canceled_to_avoid_div0(function) -> None:
    sc2.warnings.warn(f"Normalization canceled to avoid div0 error in function {function}")


def sigmoid_normalization(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh_normalization(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def max_abs_normalization(x: np.ndarray) -> np.ndarray:
    factor = x.abs().max()
    if factor == 0:
        canceled_to_avoid_div0("max_abs_normalization")
        return x
    else:
        return x / factor


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        canceled_to_avoid_div0("min_max_normalization")
        return x
    else:
        return (x - x_min) / (x_max - x_min)


def mean_std_normalization(x: np.ndarray, mean=0, std=1) -> np.ndarray:
    if std <= 0:
        canceled_to_avoid_div0("mean_std_normalization")
        return x
    else:
        return (x - mean) / std
