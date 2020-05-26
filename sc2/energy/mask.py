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
    """
    modify the pixels of energy map to -C
    thus the detector will prefer seams that go through them

    arguments:
        x:      numpy 2-d array, with shape (n, m), energy map
        mask:   numpy 2-d array, with shape (n, m)
    returns:
        y:      numpy 2-d array, with shape (n, m), modified energy map
    """

    return mask_with_constant(x, mask, constant=-1000.0)


def protect_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    modify the pixels of energy map to +inf
    thus the detector will prefer seams that not go through them

    arguments:
        x:      numpy 2-d array, with shape (n, m), energy map
        mask:   numpy 2-d array, with shape (n, m)
    returns:
        y:      numpy 2-d array, with shape (n, m), modified energy map
    """

    return mask_with_constant(x, mask, constant=np.inf)
