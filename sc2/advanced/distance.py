import numpy as np
import numba as nb
import sc2


@sc2.utils.just_in_time
def L0norm(x, y):
    return np.linalg.norm(x - y, ord=0)


@sc2.utils.just_in_time
def L1norm(x, y):
    return np.linalg.norm(x - y, ord=1)


@sc2.utils.just_in_time
def L2norm(x, y):
    return np.linalg.norm(x - y, ord=2)


@nb.jit
def euclidean(x, y): 
    return np.linalg.norm(x - y, ord=2)
