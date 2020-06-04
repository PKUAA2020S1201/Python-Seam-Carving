import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy
import sc2


def entropy_energy(image: np.ndarray) -> np.ndarray:
    x = [entropy(image[:, :, i], disk(10)) for i in range(image.shape[2])]
    x = np.sum(x, axis=0)
    x = sc2.utils.normalization.min_max_normalization(x)
    return x
