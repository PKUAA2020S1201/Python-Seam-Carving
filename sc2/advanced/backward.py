import numpy as np
import sc2


@sc2.utils.just_in_time
def seam_by_backward_energy(image: np.ndarray) -> np.ndarray:
    n, m, c = image.shape
    f = np.zeros((n, m), dtype=np.float32)
    g = np.zeros((n, m), dtype=np.int)
    seam = np.zeros(n, dtype=np.int)

    for i in range(n):
        for j in range(m):
            u, d = max(0, i - 1), min(n - 1, i + 1)
            l, r = max(0, j - 1), min(m - 1, j + 1)

            f[i, j] += np.sqrt(np.sum(np.square(image[i, l] - image[i, r])))
            f[i, j] += np.sqrt(np.sum(np.square(image[u, j] - image[d, j])))

            if i > 0:
                g[i, j] = np.argmin(f[u, l : r]) + l
                f[i, j] = f[i, j] + np.min(f[u, l : r])
            

    j = np.argmin(f[n - 1])

    for i in range(n - 1, -1, -1):
        seam[i], j = j, g[i, j]

    return seam
