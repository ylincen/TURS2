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

        self.features = X.to_numpy()
        self.target = y.to_numpy().flatten()

        self.max_rule_length = max_rule_length
        self.nrow, self.ncol = X.shape[0], X.shape[1]
        self.cached_number_of_rules_for_cl_model = 100  # for cl_model

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(self.target))

        # get_candidate_cuts (for NUMERIC only; CATEGORICAL dims will do rule.get_categorical_values)
        # self.candidate_cuts = self.get_candidate_cuts_CLASSY(num_candidate_cuts)
        # self.candidate_cuts = self.get_candidate_cuts(num_candidate_cuts)
        # self.candidate_cuts = self.get_candidate_cuts_indep_data(num_candidate_cuts)
        self.num_candidate_cuts = num_candidate_cuts
        self.candidate_cuts = self.get_candidate_cuts_quantile(num_candidate_cuts)

        self.cl_model = {}
        self.cache_cl_model()

        self.feature_names = feature_names
        self.beam_width = beam_width

    def cache_cl_model(self):
        l_number_of_variables = [universal_code_integers_maximum(n=i, maximum=self.ncol) for i in range(self.max_rule_length)]
        l_which_variables = log2comb(self.ncol, np.arange(self.max_rule_length))

        candidate_cuts_length = np.array([len(candi) for candi in self.candidate_cuts.values()], dtype=float)
        l_one_cut = np.log2(candidate_cuts_length) + 1 + 1  # 1 bit for LEFT/RIGHT, and 1 bit for one/two cuts

        l_two_cut = np.zeros(len(candidate_cuts_length))
        only_one_candi_selector = (candidate_cuts_length == 1)
        l_two_cut[only_one_candi_selector] = np.nan
        l_two_cut[~only_one_candi_selector] = np.log2(candidate_cuts_length[~only_one_candi_selector]) + \
                                              np.log2(candidate_cuts_length[~only_one_candi_selector] - 1) - np.log2(2) \
                                              + 1 # the last 1 bit is for encoding one/two cuts
        l_cut = np.array([l_one_cut, l_two_cut])

        l_number_of_rules = [universal_code_integers(i) for i in range(self.cached_number_of_rules_for_cl_model)]

        self.cl_model["l_number_of_variables"] = l_number_of_variables
        self.cl_model["l_cut"] = l_cut
        self.cl_model["l_which_variables"] = l_which_variables
        self.cl_model["l_number_of_rules"] = l_number_of_rules

    def get_candidate_cuts_CLASSY(self, num_candidate_cuts):
        candidate_cuts = {}

        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                quantile_percentage = [1 / (num_candidate_cuts_i + 1) * ncut for ncut in range(0, num_candidate_cuts_i + 2)]
                value_quantiles = np.nanquantile(feature, quantile_percentage, interpolation='midpoint')[1:-1]
                value_quantiles = np.unique(value_quantiles)
                candidate_cuts[i] = value_quantiles
        return candidate_cuts

    def get_candidate_cuts_quantile(self, num_candidate_cuts):
        candidate_cuts = {}

        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                quantile_percentage = np.linspace(0, 1, num_candidate_cuts_i + 2)[1:-1]
                candidate_cuts[i] = np.quantile(feature, quantile_percentage)
        return candidate_cuts

    def get_candidate_cuts_indep_data(self, num_candidate_cuts):
        candidate_cuts = {}
        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                candidate_cuts[i] = np.linspace(unique_value[0], unique_value[-1], num_candidate_cuts_i)
        return candidate_cuts

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
