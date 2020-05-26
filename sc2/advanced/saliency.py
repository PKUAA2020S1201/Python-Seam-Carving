import numpy as np
import cv2
import sc2


def static_saliency(image: np.ndarray) -> np.ndarray:
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return sc2.utils.min_max_normalization(saliencyMap)