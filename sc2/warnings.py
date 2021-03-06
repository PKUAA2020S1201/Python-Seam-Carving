import warnings


class SeamCarvingWarning(Warning):
    pass


def warn(*args, **kwargs) -> None:
    """
    raise SeamCarvingWarning
    """

    warnings.warn(*args, **kwargs, category=SeamCarvingWarning)


def ignore_numba_warnings():
    """
    ignore numba warnings
    """

    from numba import NumbaWarning, NumbaDeprecationWarning
    from warnings import simplefilter

    simplefilter("ignore", NumbaWarning)


def ignore_seamcarving_warnings():
    """
    ignore sc2 warnings
    """

    from warnings import simplefilter
    from sc2.warnings import SeamCarvingWarning

    simplefilter("ignore", SeamCarvingWarning)
