import cv2
import numpy as np
import numba as nb
import seamcarving as sc


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def canny_edges(image):
    return cv2.Canny(convert_to_gray(image), 50, 150, apertureSize=3)


def sobel_detection(image):
    gray = convert_to_gray(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    result = np.abs(sobelx) + np.abs(sobely)
    result[result < 100] = 0
    result[result > 255] = 255
    return sc.normalized(result)


def hough_detection(image, threshold=20):
    edges = canny_edges(image)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold,
        minLineLength=10,
        maxLineGap=200
    )
    result = np.zeros((image.shape[0], image.shape[1]))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (1,), 2)
    return sc.normalized(result)


@nb.jit
def seam_to_mask(shape, seam):
    n = shape[0]
    m = shape[1]
    result = np.zeros(shape)
    for i in range(n):
        result[i, seam[i]] = 1
    return result


@nb.jit
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


@nb.jit
def diffusion_energy(energy, seam, p=0.5):
    return energy + diffusion_delta(energy, seam, p)


@nb.jit
def diffusion_delta(energy, seam, p=0.5):
    n, m = energy.shape
    delta = np.zeros_like(energy)
    for i in range(n):
        j = seam[i]
        if j > 0:
            delta[i, j - 1] += p * energy[i, j]
        if j < m - 1:
            delta[i, j + 1] += p * energy[i, j]
    return delta


def static_saliency(image):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return saliencyMap
