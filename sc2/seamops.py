import numpy as np
import sc2


@sc2.utils.just_in_time
def detect_seam(energy: np.ndarray) -> np.ndarray:
    n, m = energy.shape
    f = np.copy(energy)
    g = np.zeros((n, m), dtype=np.int)
    seam = np.zeros(n, dtype=np.int)

    for i in range(1, n):
        for j in range(0, m):
            l = max(0, j - 1)
            r = min(m, j + 1)

            g[i, j] = np.argmin(f[i - 1, l : r]) + l
            f[i, j] = f[i, j] + np.min(f[i - 1, l : r])

    
    j = np.argmin(f[n - 1])

    for i in range(n - 1, -1, -1):
        seam[i], j = j, g[i, j]
    
    return seam


@sc2.utils.just_in_time
def remove_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    n = image.shape[0]
    m = image.shape[1]

    if len(image.shape) == 2:
        result = np.zeros((n, m - 1), dtype=image.dtype)
    elif len(image.shape) == 3:
        result = np.zeros((n, m - 1, image.shape[2]), dtype=image.dtype)
    # else:
    #     raise NotImplementedError(f"remove_seam not implemented for image with shape {image.shape}")

    for i in range(n):
        j = seam[i]
        part_a = image[i, :j]
        part_b = image[i, j + 1:]
        result[i] = np.concatenate([part_a, part_b], axis=0)
    
    return result


@sc2.utils.just_in_time
def expend_seam(image: np.ndarray, seam : np.ndarray) -> np.ndarray:
    n = image.shape[0]
    m = image.shape[1]

    if len(image.shape) == 2:
        result = np.zeros((n, m + 1), dtype=image.dtype)
    elif len(image.shape) == 3:
        result = np.zeros((n, m + 1, image.shape[2]), dtype=image.dtype)
    # else:
    #     raise NotImplementedError(f"expend_seam not implemented for image with shape {image.shape}")
    
    for i in range(n):
        j = seam[i]
        if 0 < j < m - 1:
            result[i, :j] = image[i, :j]
            result[i, j + 2:] = image[i, j + 1:]
            result[i, j] = image[i, j - 1]
            result[i, j + 1] = image[i, j]
        elif j == 0:
            result[i, 1:] = image[i, :]
            result[i, 0] = image[i, 0]
        else:
            result[i, :m] = image[i, :]
            result[i, m] = image[i, m - 1]
    return result


@sc2.utils.just_in_time
def relocate_seams(seams):
    n, m = seams.shape

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(m):
                if seams[i][k] <= seams[j][k]:
                    seams[j][k] += 2

    return seams


@sc2.utils.just_in_time
def image_with_seam(image: np.ndarray, seam: np.ndarray, pixel=[255, 255, 255]) -> np.ndarray:
    n, m, c = image.shape
    pixel = np.array(pixel)
    result = np.copy(image)

    for i in range(n):
        result[i, seam[i]] = pixel
    
    return result
