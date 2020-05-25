import numpy as np
import cv2
import sc2


@sc2.utils.just_in_time
def seam_to_mask(shape, seam):
    n = shape[0]
    m = shape[1]
    result = np.zeros(shape)
    for i in range(n):
        result[i, seam[i]] = 1
    return result


@sc2.utils.just_in_time
def pixels_not_zero(src):
    n = src.shape[0]
    m = src.shape[1]
    result = list()
    for i in range(n):
        for j in range(m):
            if src[i, j] != 0:
                result.append((i, j))
    return result


def circles(src, r=5):
    n = src.shape[0]
    m = src.shape[1]
    result = np.zeros((n, m))
    for i, j in pixels_not_zero(src):
        cv2.circle(result, (j, i), r, 1, -1)
    return result
