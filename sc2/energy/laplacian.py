import numpy as np
import cv2


def laplacian(image: np.ndarray, ddepth=cv2.CV_64F, ksize=None):
    return np.sum(np.abs(cv2.Laplacian(image, ddepth, ksize)), axis=-1)
