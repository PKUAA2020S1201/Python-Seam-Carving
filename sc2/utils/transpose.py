import numpy as np


def transpose(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return np.transpose(image, [1, 0])
    elif len(image.shape) == 3:
        return np.transpose(image, [1, 0, 2])
    else:
        raise NotImplementedError(
            f"transpose not implemented for image with shape {image.shape}"
        )
