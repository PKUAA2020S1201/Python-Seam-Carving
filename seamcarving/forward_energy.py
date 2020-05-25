import numpy as np
import numba as nb
import cv2


@nb.jit
def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


@nb.jit
def seam_with_minimum_forward_energy(image):
    n, m, c = image.shape
    f = np.zeros((n, m), dtype=np.float32)
    g = np.zeros((n, m), dtype=np.int)
    for j in range(1, m - 1):
        if j == 0:
            delta = distance(image[0, j + 1], image[0, j]) * 2
        elif j == m - 1:
            delta = distance(image[0, j - 1], image[0, j]) * 2
        else:
            delta = distance(image[0, j - 1], image[0, j + 1])
        f[0, j] = delta
    for i in range(1, n):
        for j in range(m):
            tmp = np.zeros(3, dtype=np.float32)
            tmp[1] = f[i - 1, j]
            tmp[0] = np.inf if j == 0 else f[i - 1, j - 1] + np.sum(np.abs(image[i, j - 1] - image[i - 1, j]))
            tmp[2] = np.inf if j == m - 1 else f[i - 1, j + 1] + np.sum(np.abs(image[i, j + 1] - image[i - 1, j]))
            if j == 0:
                delta = distance(image[i, j + 1], image[i, j]) * 2
            elif j == m - 1:
                delta = distance(image[i, j - 1], image[i, j]) * 2
            else:
                delta = distance(image[i, j - 1], image[i, j + 1])
            g[i, j] = j - 1 + np.argmin(tmp)
            f[i, j] = np.min(tmp) + delta
    seam = np.zeros(n, dtype=np.int)
    j = np.argmin(f[n - 1])
    for i in range(n - 1, 0, -1):
        seam[i], j = j, g[i, j]
    return seam