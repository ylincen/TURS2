import sys

import numpy as np
import math
from utils_modelencoding import *


class ModelEncodingDependingOnData:
    def __init__(self, data_info, given_ncol=None):
        self.cached_cl_model = {}
        self.max_num_rules = 100  # an upper bound for the number of rules, just for cl_model caching
        self.data_info = data_info
        if given_ncol is None:
            self.data_ncol_for_encoding = data_info.ncol
            self.cache_cl_model(data_info.ncol, data_info.max_rule_length, data_info.candidate_cuts)
        else:
            self.data_ncol_for_encoding = given_ncol
            self.cache_cl_model(given_ncol, data_info.max_rule_length, data_info.candidate_cuts)


    def cache_cl_model(self, data_ncol, max_rule_length, candidate_cuts):
        l_number_of_variables = [universal_code_integers(i) for i in range(max(data_ncol-1, max_rule_length))]
        l_which_variables = log2comb(data_ncol, np.arange(max(data_ncol-1, max_rule_length)))

        candidate_cuts_length = np.array([len(candi) for candi in candidate_cuts.values()], dtype=float)
        only_one_candi_selector = (candidate_cuts_length == 1)
        zero_candi_selector = (candidate_cuts_length == 0)

        l_one_cut = np.zeros(len(candidate_cuts_length), dtype=float)
        l_one_cut[~zero_candi_selector] = np.log2(candidate_cuts_length[~zero_candi_selector]) + 1 + 1  # 1 bit for LEFT/RIGHT, and 1 bit for one/two cuts
        l_one_cut[only_one_candi_selector] = l_one_cut[only_one_candi_selector] - 1

        l_two_cut = np.zeros(len(candidate_cuts_length))
        l_two_cut[only_one_candi_selector] = np.nan
        two_candi_selector = (candidate_cuts_length > 1)

        # TODO: reconsider the l_two_cut from the perspective of hypothesis testing
        l_two_cut[two_candi_selector] = np.log2(candidate_cuts_length[two_candi_selector]) + \
                                        np.log2(candidate_cuts_length[two_candi_selector] - 1) - np.log2(2) \
                                        + 1  # the last 1 bit is for encoding one/two cuts
        # l_two_cut[two_candi_selector] = np.log2(candidate_cuts_length[two_candi_selector]) + \
        #                                 np.log2(candidate_cuts_length[two_candi_selector] - 1) + 1 + 2
                                        # the 1 bit is for encoding one/two cuts, and 2 bits for left/right for each cut
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

    # TODO: below: cl model depending on data; above: vanilla definition of rule_cl_model
    def rule_cl_model_dep(self, condition_matrix, col_orders):
        condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        num_variables = np.count_nonzero(condition_count)
        l_num_variables = self.cached_cl_model["l_number_of_variables"][num_variables]
        l_which_variables = self.cached_cl_model["l_which_variables"][num_variables]

        bool_ = np.ones(len(self.data_info.features), dtype=bool)
        l_cuts = 0
        for index, col in enumerate(col_orders):
            up_bound, low_bound = np.max(self.data_info.features[bool_, col]), np.min(self.data_info.features[bool_, col])
            num_cuts = np.count_nonzero((self.data_info.candidate_cuts[col] >= low_bound) &
                                        (self.data_info.candidate_cuts[col] <= up_bound))
            if condition_count[col] == 1:
                l_cuts += np.log2(num_cuts)
            else:
                assert num_cuts >= 2
                l_cuts += np.log2(num_cuts) + np.log2(num_cuts - 1) - np.log2(2)

            if index != len(col_orders) - 1:
                assert condition_count[col] == 1 or condition_count[col] == 2
                if condition_count[col] == 1:
                    if not np.isnan(condition_matrix[0, col]):
                        bool_ = bool_ & (self.data_info.features[:, col] <= condition_matrix[0, col])
                    else:
                        bool_ = bool_ & (self.data_info.features[:, col] > condition_matrix[1, col])
                else:
                    bool_ = bool_ & ((self.data_info.features[:, col] <= condition_matrix[0, col]) &
                                     (self.data_info.features[:, col] > condition_matrix[1, col]))

        return l_num_variables + l_which_variables + l_cuts

    def cl_model_after_growing_rule_on_icol(self, rule, ruleset, icol, cut_option):
        # TODO: I think this block is not used anymore, but let's see what happens when tests on more datasets
        # TODO: i.e., "if rule is None" will never be true;
        if rule is None:  # calculate the cl_model when no new rule is added to ruleset
            l_num_rules = universal_code_integers(len(ruleset.rules))

            # TODO: don't forget here
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1) / np.log(2)
            # cl_redundancy_rule_orders = 0

            return l_num_rules + ruleset.allrules_cl_model - cl_redundancy_rule_orders

        condition_count = np.array(rule.condition_count)
        icols_in_order = rule.icols_in_order
        condition_matrix = np.array(rule.condition_matrix)

        # when the rule is still being grown by adding condition using icol, we need to update the condition_count;
        if icol is not None and cut_option is not None:
            # if rule.condition_matrix[cut_option, icol] is np.nan:  # HERE why it is 0??
            if np.isnan(rule.condition_matrix[cut_option, icol]):
                condition_count[icol] += 1
                condition_matrix[0, icol] = np.inf  # TODO: Note that this is just a place holder, to make this position not equal to np.nan; Make it better later.

            if icol not in icols_in_order:
                icols_in_order = icols_in_order + [icol]

        # TODO: note that here is a choice based on the assumption that we can use $X$ to encode the model;
        # cl_model_rule_after_growing = self.rule_cl_model(condition_count)
        cl_model_rule_after_growing = self.rule_cl_model_dep(condition_matrix, icols_in_order)

        l_num_rules = universal_code_integers(len(ruleset.rules) + 1)

        # Remove some redundancy by the fact that the order of rules does not matter
        cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 2) / np.log(2)
        # cl_redundancy_rule_orders = 0

        return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset.allrules_cl_model


