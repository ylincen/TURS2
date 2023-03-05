from scipy.special import gammaln, comb
import numpy as np
from math import log


def log2comb(n, k):
    return (gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)) / log(2)


def universal_code_integers_maximum(n: int, maximum : int) -> float:
    """ computes the universal code of integers when there is a known maximum integer
    This is equivalent to applying the maximum entropy principle knowing the maximum,
    and it equalitarian  division of the non-used probability (the ones after the maximum)
    by all the used number (1 until maximum).
    """
    probability_until_max = np.sum([2**-universal_code_integers(n_aux) for n_aux in range(1, maximum+1)])
    probability_left = 1 - probability_until_max
    probability_n = 2**-universal_code_integers(n) + probability_left/maximum
    logsum = -np.log2(probability_n)
    return logsum


def universal_code_integers(value: int) -> float:
    """ computes the universal code of integers
    """
    const = 2.865064
    logsum = np.log2(const)
    if value == 0:
        logsum = 0
    elif value > 0:
        while True: # Recursive log
            value = np.log2(value)
            if value < 0.001:
                break
            logsum += value
    elif value < 0:
        raise ValueError('n should be larger than 0. The value was: {}'.format(value))
    return logsum