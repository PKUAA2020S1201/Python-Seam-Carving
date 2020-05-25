import warnings


class SeamCarvingWarning(Warning):
    pass


def warn(*args, **kwargs) -> None:
    warnings.warn(*args, **kwargs, category=SeamCarvingWarning)


def ignore_numba_warnings():
    from numba import NumbaWarning
    from warnings import simplefilter

    simplefilter("ignore", NumbaWarning)


def ignore_seamcarving_warnings():
    from warnings import simplefilter
    from sc2.warnings import SeamCarvingWarning

    simplefilter("ignore", SeamCarvingWarning)
