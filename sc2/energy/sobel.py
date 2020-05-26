import numpy as np
import cv2
import sc2


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), 0)


def sobel(image: np.ndarray) -> np.ndarray:
    """
    calculate sobel gradient as energy map

    arguments:
        image:  numpy 2-d or 3-d array, with shape (n, m, ...)
    returns:
        energy: numpy 2-d array, with shape (n, m)
    """

    image = np.copy(image)
    image = convert_to_gray(image)
    image = gaussian_blur(image)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    result = np.abs(sobelx) + np.abs(sobely)
    return sc2.utils.min_max_normalization(result)
