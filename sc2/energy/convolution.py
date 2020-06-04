import numpy as np
import sc2


"""
NOTE: convolution.py

    in this file, we implemeted a full-python gradient calculator
    depends on only numpy and nothing else.
    first, we implemented convolution2D in python.
    second, we calculate gradient using conv2d and sobel kernel.
    the function can be faster if jit is enabled, 
    but we still highly recommand you to use opencv based energy function.
"""


@sc2.utils.just_in_time
def conv2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n, m, c = x.shape
    h, w = kernel.shape
    kernel = np.stack([kernel] * c, axis=2)
    padding = [
        [h // 2] * 2,
        [w // 2] * 2,
        [0] * 2
    ]
    y = np.zeros_like(x, dtype=np.float32)
    x = np.pad(x, padding, mode="edge")
    for i in range(n):
        for j in range(m):
            clip = x[i : i + h, j : j + w, ...]
            conv = clip * kernel
            conv = np.sum(conv, axis=(0, 1))
            y[i, j] = conv
    return y


def convolution_gradient(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    kernel_y = np.transpose(kernel_x, (1, 0))
    dx = np.abs(conv2d(image, kernel_x))
    dy = np.abs(conv2d(image, kernel_y))
    g = np.power(np.power(dx, 2) + np.power(dy, 2), 1 / 2)
    g = np.sum(g, axis=-1)
    g = sc2.utils.normalization.min_max_normalization(g)
    return g
