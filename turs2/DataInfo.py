import sys
import pandas as pd
import numpy as np
from constant import *
from math import log
from scipy.special import gammaln, comb


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


class DataInfo:
    def __init__(self, X, y, num_candidate_cuts, max_rule_length, feature_names, beam_width):
        """
        Meta-data for an input data
        data: pandas data frame
        features: feature matrix in numpy nd array
        target: target variable in numpy 1d array
        """

        self.features = X
        self.target = y

        self.max_rule_length = max_rule_length
        self.nrow, self.ncol = X.shape[0], X.shapep[1]

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(self.target))

        # get_candidate_cuts (for NUMERIC only; CATEGORICAL dims will do rule.get_categorical_values)
        self.candidate_cuts = self.get_candidate_cuts(num_candidate_cuts)

        self.cl_model = {}
        self.feature_names = feature_names
        self.beam_width = beam_width

    def cache_cl_model(self):
        l_number_of_variables = [universal_code_integers_maximum(n=i+1, maximum=self.ncol) for i in range(self.ncol)]
        l_which_variables = log2comb(self.ncol, np.arange(self.max_rule_length) + 1)

        candidate_cuts_length = np.array([len(candi) for candi in self.candidate_cuts.values()], dtype=float)
        l_one_cut = np.log2(candidate_cuts_length)
        l_two_cut = np.log2(candidate_cuts_length) * 2
        l_cut = np.array(l_one_cut, l_two_cut)

        self.cl_model["l_number_of_variables"] = l_number_of_variables
        self.cl_model["l_cut"] = l_cut
        self.cl_model["l_which_variables"] = l_which_variables

    def get_candidate_cuts(self, num_candidate_cuts):
        candidate_cuts = {}
        dim_iter_counter = -1

        if num_candidate_cuts is list:
            pass
        else:
            num_candidate_cuts = np.repeat(num_candidate_cuts, self.ncol)

        for i, feature in enumerate(self.features.T):
            dim_iter_counter += 1

            sort_feature = np.unique(feature)
            if len(sort_feature) <= 1:
                candidate_cut_this_dimension = np.array([], dtype=float)
                candidate_cuts[i] = candidate_cut_this_dimension
            else:
                candidate_cut_this_dimension = \
                    (sort_feature[0:(len(sort_feature) - 1)] + sort_feature[1:len(sort_feature)]) / 2
                # to set the bins for each numeric dimension
                if (num_candidate_cuts[i] > 1) & (len(candidate_cut_this_dimension) > num_candidate_cuts[i]):
                    select_indices = np.linspace(0, len(candidate_cut_this_dimension) - 1, num_candidate_cuts[i] + 1,
                                                 endpoint=True, dtype=int)
                    select_indices = select_indices[
                                     1:(len(select_indices) - 1)]  # remove the start and end point
                    candidate_cuts[i] = candidate_cut_this_dimension[select_indices]
                else:
                    candidate_cuts[i] = candidate_cut_this_dimension

        return candidate_cuts
