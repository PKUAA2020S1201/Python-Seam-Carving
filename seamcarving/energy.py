import numpy as np
import numba as nb
import cv2


@nb.jit
def gradient_energy(image):
    n, m, c = image.shape
    padding = (
        (1, 1),
        (1, 1),
        (0, 0)
    )
    kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    kernel_y = kernel_x.transpose((1, 0))
    kernel_x = np.stack([kernel_x] * c, axis=2)
    kernel_y = np.stack([kernel_y] * c, axis=2)
    def convolution(x, kernel):
        y = np.zeros_like(x)
        x = np.pad(x, padding, mode="edge")
        for i in range(n):
            for j in range(m):
                clip = x[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                conv = np.sum(clip * kernel, axis=(0, 1))
                y[i, j] = conv
        return y
    dx = np.abs(convolution(image, kernel_x))
    dy = np.abs(convolution(image, kernel_y))
    gradient = np.power(np.power(dx, 2) + np.power(dy, 2), 0.5)
    gradient = np.sum(gradient, axis=-1)
    return gradient


def laplacian_energy(image, ddepth=cv2.CV_64F, ksize=None):
    return np.sum(np.abs(cv2.Laplacian(image, ddepth, ksize)), axis=-1)
