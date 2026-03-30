import math

import numpy as np
from utils_modelencoding import *


class ModelEncodingDependingOnData:
    def __init__(self, data_info, given_ncol=None):
        self.cached_cl_model = {}
        self.max_num_rules = 100  # upper bound for number of rules, used for caching
        self.data_info = data_info
        if given_ncol is None:
            self.data_ncol_for_encoding = data_info.ncol
            self.cache_cl_model(data_info.ncol, data_info.max_grow_iter, data_info.candidate_cuts)
        else:
            self.data_ncol_for_encoding = given_ncol
            self.cache_cl_model(given_ncol, data_info.max_grow_iter, data_info.candidate_cuts)

    def cache_cl_model(self, data_ncol, max_rule_length, candidate_cuts):
        l_number_of_variables = [universal_code_integers(i) for i in range(max(data_ncol-1, max_rule_length))]
        l_which_variables = log2comb(data_ncol, np.arange(max(data_ncol-1, max_rule_length)))

        candidate_cuts_length = np.array([len(candi) for candi in candidate_cuts.values()], dtype=float)
        only_one_candi_selector = (candidate_cuts_length == 1)
        zero_candi_selector = (candidate_cuts_length == 0)

        # 1 bit for LEFT/RIGHT direction, 1 bit for one/two cuts
        l_one_cut = np.zeros(len(candidate_cuts_length), dtype=float)
        l_one_cut[~zero_candi_selector] = np.log2(candidate_cuts_length[~zero_candi_selector]) + 1 + 1
        l_one_cut[only_one_candi_selector] = l_one_cut[only_one_candi_selector] - 1

        # Choose 2 unordered cuts from n: log2(n) + log2(n-1) - log2(2); +1 bit for one/two cuts
        l_two_cut = np.zeros(len(candidate_cuts_length))
        l_two_cut[only_one_candi_selector] = np.nan
        two_candi_selector = (candidate_cuts_length > 1)
        l_two_cut[two_candi_selector] = (np.log2(candidate_cuts_length[two_candi_selector]) +
                                         np.log2(candidate_cuts_length[two_candi_selector] - 1) -
                                         np.log2(2) + 1)
        l_cut = np.array([l_one_cut, l_two_cut])

        l_number_of_rules = [universal_code_integers(i) for i in range(self.max_num_rules)]

        self.cached_cl_model["l_number_of_variables"] = l_number_of_variables
        self.cached_cl_model["l_cut"] = l_cut
        self.cached_cl_model["l_which_variables"] = l_which_variables
        self.cached_cl_model["l_number_of_rules"] = l_number_of_rules

    def rule_cl_model(self, condition_count):
        num_variables = np.count_nonzero(condition_count)
        l_num_variables = self.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = self.cached_cl_model["l_which_variables"][num_variables]
        l_cuts = (
                np.sum(self.cached_cl_model["l_cut"][0][condition_count == 1]) +
                np.sum(self.cached_cl_model["l_cut"][1][condition_count == 2])
        )
        return l_num_variables + l_which_variables + l_cuts

    def rule_cl_model_dep(self, condition_matrix, col_orders):
        """Data-dependent rule code length: number of candidate cuts is counted
        within the data subspace reached by preceding conditions."""
        condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        num_variables = np.count_nonzero(condition_count)
        l_num_variables = self.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = self.cached_cl_model["l_which_variables"][num_variables]

        bool_ = np.ones(len(self.data_info.features), dtype=bool)
        l_cuts = 0.0
        for index, col in enumerate(col_orders):
            if np.count_nonzero(bool_) == 0:
                l_cuts += 10**8
            else:
                up_bound = np.max(self.data_info.features[bool_, col])
                low_bound = np.min(self.data_info.features[bool_, col])
                num_cuts = np.count_nonzero((self.data_info.candidate_cuts[col] >= low_bound) &
                                            (self.data_info.candidate_cuts[col] <= up_bound))
                if condition_count[col] == 1:
                    l_cuts += np.log2(num_cuts) if num_cuts > 0 else 0
                else:
                    l_cuts += (np.log2(num_cuts) + np.log2(num_cuts - 1) - np.log2(2)) if num_cuts >= 2 else 0

                if index != len(col_orders) - 1:
                    assert condition_count[col] == 1 or condition_count[col] == 2
                    if condition_count[col] == 1:
                        if not np.isnan(condition_matrix[0, col]):
                            bool_ = bool_ & (self.data_info.features[:, col] < condition_matrix[0, col])
                        else:
                            bool_ = bool_ & (self.data_info.features[:, col] >= condition_matrix[1, col])
                    else:
                        bool_ = bool_ & ((self.data_info.features[:, col] < condition_matrix[0, col]) &
                                         (self.data_info.features[:, col] >= condition_matrix[1, col]))

        return l_num_variables + l_which_variables + l_cuts

    def cl_model_after_growing_rule_on_icol(self, rule, ruleset, icol, cut_option):
        if rule is None:
            l_num_rules = universal_code_integers(len(ruleset.rules))
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1) / np.log(2)
            return l_num_rules + ruleset.allrules_cl_model - cl_redundancy_rule_orders

        condition_count = np.array(rule.condition_count)
        icols_in_order = rule.icols_in_order
        condition_matrix = np.array(rule.condition_matrix)

        if icol is not None and cut_option is not None:
            if np.isnan(rule.condition_matrix[cut_option, icol]):
                condition_count[icol] += 1
                condition_matrix[0, icol] = np.inf  # placeholder: marks position as non-nan
            if icol not in icols_in_order:
                icols_in_order = icols_in_order + [icol]

        cl_model_rule_after_growing = self.rule_cl_model_dep(condition_matrix, icols_in_order)

        if icol is None and cut_option is None:
            l_num_rules = universal_code_integers(len(ruleset.rules))
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1) / np.log(2)
        else:
            l_num_rules = universal_code_integers(len(ruleset.rules) + 1)
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 2) / np.log(2)

        return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset.allrules_cl_model
