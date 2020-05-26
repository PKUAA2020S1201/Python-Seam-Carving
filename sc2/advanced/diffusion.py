import numpy as np
import sc2


@sc2.utils.just_in_time
def diffusion(x: np.ndarray, seam: np.ndarray, p=0.5) -> np.ndarray:
    n, m = x.shape[:2]
    result = np.zeros_like(x)
    for i in range(n):
        j = seam[i]
        if j > 0:
            result[i, j - 1] = p * x[i, j]
        if j < m - 1:
            result[i, j + 1] = p * x[i, j]
    return result
