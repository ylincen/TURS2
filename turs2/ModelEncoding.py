import numpy as np
import math
from utils_modelencoding import *


class ModelEncodingDependingOnData:
    def __init__(self, data_info, given_ncol=None):
        self.cached_cl_model = {}
        self.max_num_rules = 100  # an upper bound for the number of rules, just for cl_model caching
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


    def cl_model_after_growing_rule_on_icol(self, rule, ruleset, icol, cut_option):
        if rule is None:  # calculate the cl_model when no new rule is added to ruleset
            l_num_rules = universal_code_integers(len(ruleset.rules))

            # TODO: don't forget here
            cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 1) / np.log(2)
            # cl_redundancy_rule_orders = 0

            return l_num_rules + ruleset.allrules_cl_model - cl_redundancy_rule_orders

        condition_count = np.array(rule.condition_count)

        # when the rule is still being grown by adding condition using icol, we need to update the condition_count;
        if icol is not None and cut_option is not None:
            # if rule.condition_matrix[cut_option, icol] is np.nan:  # HERE why it is 0??
            if np.isnan(rule.condition_matrix[cut_option, icol]):
                condition_count[icol] += 1

        # cl_model_rule_after_growing = self.rule_cl_model(rule.condition_count)
        cl_model_rule_after_growing = self.rule_cl_model(condition_count)

        l_num_rules = universal_code_integers(len(ruleset.rules) + 1)
        cl_redundancy_rule_orders = math.lgamma(len(ruleset.rules) + 2) / np.log(2)
        # cl_redundancy_rule_orders = 0

        return l_num_rules + cl_model_rule_after_growing - cl_redundancy_rule_orders + ruleset.allrules_cl_model


