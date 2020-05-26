import numpy as np
import numba as nb
import sc2


@sc2.utils.just_in_time
def seam_by_forward_energy(image: np.ndarray) -> np.ndarray:
    n, m, c = image.shape
    image = image.astype(np.float32)
    f = np.zeros((n, m), dtype=np.float32)
    g = np.zeros((n, m), dtype=np.int)
    seam = np.zeros(n, dtype=np.int)

    for i in range(n):
        for j in range(m):
            u, d = max(0, i - 1), min(n - 1, i + 1)
            l, r = max(0, j - 1), min(m - 1, j + 1)

            f[i, j] = np.sqrt(np.sum(np.square(image[i, l] - image[i, r])))
            
            if i > 0:
                tmp = np.array([
                    f[u, l] + np.sqrt(np.sum(np.square(image[i, l] - image[u, j]))),
                    f[u, j],
                    f[u, r] + np.sqrt(np.sum(np.square(image[i, r] - image[u, j])))
                ])
                g[i, j] = j - 1 + np.argmin(tmp)
                f[i, j] = f[i, j] + np.min(tmp)

    j = np.argmin(f[n - 1])

    for i in range(n - 1, -1, -1):
        seam[i], j = j, g[i, j]

    return seam
