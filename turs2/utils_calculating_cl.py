from numba import njit
import numpy as np
import scipy.special
from constant import *

@njit
def calc_probs(target, num_class, smoothed=False):
    counts = np.bincount(target, minlength=num_class)
    if smoothed:
        counts = counts + np.ones(num_class, dtype='int64')
    if np.sum(counts) == 0:
        return counts / 1.0
    else:
        return counts / np.sum(counts)