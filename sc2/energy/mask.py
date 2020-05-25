import numpy as np
import sc2


@sc2.utils.just_in_time
def mask_with_constant(
    x: np.ndarray,
    mask: np.ndarray,
    constant=0
) -> np.ndarray:
    x = np.copy(x)
    m = mask != 0
    x[m] = constant
    return x


def remove_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask_with_constant(x, mask, constant=-1000.0)


def protect_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask_with_constant(x, mask, constant=np.inf)
