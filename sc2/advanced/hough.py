import numpy as np
import cv2
import sc2


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def canny_edges(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(convert_to_gray(image), 50, 150, apertureSize=3)


def hough(image: np.ndarray, threshold=20, minlen=10, maxgap=200) -> np.ndarray:
    image = np.copy(image)
    image = canny_edges(image)
    lines = cv2.HoughLinesP(
        image, 1, np.pi / 180, threshold,
        minLineLength=minlen,
        maxLineGap=maxgap
    )
    result = np.zeros((image.shape[0], image.shape[1]))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (1,), 2)
    return sc2.utils.min_max_normalization(result)