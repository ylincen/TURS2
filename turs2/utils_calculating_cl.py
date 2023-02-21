from numba import njit
import numpy as np
import scipy.special
from constant import *
import random

@njit
def calc_probs(target, num_class, smoothed=False):
    counts = np.bincount(target, minlength=num_class)
    if smoothed:
        counts = counts + np.ones(num_class, dtype='int64')
    if np.sum(counts) == 0:
        return counts / 1.0
    else:
        return counts / np.sum(counts)


@njit
def calc_prequential(target, num_class, num_rep=1):
    negloglike = 0

    for iter in range(num_rep):
        # tt = np.array(random.sample( list(target), len(target)))
        indices = np.random.choice(np.arange(len(target)), replace=False, size=len(target))
        tt = target[indices]
        counts = np.zeros((num_class, len(target)), dtype="int64")
        for i in range(num_class):
            counts[i] = np.cumsum(tt == i) + 1
            probs = counts[i] / (np.arange(len(target)) + 1 + num_class)
            probs[1:] = probs[:-1]
            probs[0] = 1 / num_class
            negloglike_i = -np.sum(np.log2(probs[tt == i]))
            negloglike += negloglike_i

    return negloglike / num_rep
