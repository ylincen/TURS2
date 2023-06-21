from numba import njit
import numpy as np
import scipy.special
from constant import *
import random
from nml_regret import *


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
def calc_prequential(target, num_class, num_rep=1, init_points=None):
    negloglike = 0

    if init_points is None:
        init_points = np.ones(num_class, dtype="int64")

    for iter in range(num_rep):
        # tt = np.array(random.sample( list(target), len(target)))
        indices = np.random.choice(np.arange(len(target)), replace=False, size=len(target))
        tt = target[indices]
        counts = np.zeros((num_class, len(target)), dtype="int64")
        for i in range(num_class):
            counts[i] = np.cumsum(tt == i) + init_points[i]
            probs = counts[i] / (np.arange(len(target)) + 1 + np.sum(init_points))
            probs[1:] = probs[:-1]
            probs[0] = 1 / num_class
            negloglike_i = -np.sum(np.log2(probs[tt == i]))
            negloglike += negloglike_i

    return negloglike / num_rep


def calc_negloglike(p, n):
    return -n * np.sum(np.log2(p[p !=0 ]) * p[p != 0])


def check_validity_growth(r1, r2):
    # assume r1 fully cover r2;
    n1 = r1.coverage
    n2 = r2.coverage
    data_info_ = r1.data_info
    p1 = calc_probs(data_info_.target[r1.indices], data_info_.num_class)
    p2 = calc_probs(data_info_.target[r2.indices], data_info_.num_class)
    p3 = calc_probs(data_info_.target[r1.bool_array & ~r2.bool_array], data_info_.num_class)

    nll1 = calc_negloglike(p1, n1)
    nll2 = calc_negloglike(p2, n2)
    nll3 = calc_negloglike(p3, n1-n2)
    validity = nll1 + regret(n1, 2) + r1.cl_model - nll2 - nll3 - regret(n2, 2) - regret(n1-n2, 2) - r2.cl_model
    return validity


