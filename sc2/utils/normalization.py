"""
sc2.utils.normalization

many normalization methods
"""

import numpy as np
import sc2


def canceled_to_avoid_div0(function) -> None:
    sc2.warnings.warn(f"Normalization canceled to avoid div0 error in function {function}")


def sigmoid_normalization(x: np.ndarray) -> np.ndarray:
    """
    sigmoid normalization

    formula:
        y = 1 / (1 + exp(-x))
    """

    return 1 / (1 + np.exp(-x))


def tanh_normalization(x: np.ndarray) -> np.ndarray:
    """
    tanh normalization

    formula:
        y = sinh(x) / cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    return np.tanh(x)


def max_abs_normalization(x: np.ndarray) -> np.ndarray:
    """
    max abs normalization

    formula:
        y = x / max(|x|)
    """
    factor = x.abs().max()
    if factor == 0:
        canceled_to_avoid_div0("max_abs_normalization")
        return x
    else:
        return x / factor


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """
    min max normalization

    formula:
        y = (x - min(x)) / (max(x) - min(x))
    """

    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        canceled_to_avoid_div0("min_max_normalization")
        return x
    else:
        return (x - x_min) / (x_max - x_min)


def mean_std_normalization(x: np.ndarray, mean=0, std=1) -> np.ndarray:
    """
    mean std normalization

    formula:
        y = (x - mean) / std
    """

    if std <= 0:
        canceled_to_avoid_div0("mean_std_normalization")
        return x
    else:
        return (x - mean) / std
