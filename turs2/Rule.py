import numpy as np
from turs2.utils_calculating_cl import *
from turs2.nml_regret import *
from functools import partial


def calc_negloglike(p, n):
    return -n * np.sum(np.log2(p[p !=0 ]) * p[p != 0])


class Rule:
    def __init__(self, indices, indices_excl_overlap, data_info, rule_base, condition_matrix, ruleset):
        self.ruleset = ruleset

        self.indices = indices  # corresponds to the original dataset
        self.indices_excl_overlap = indices_excl_overlap  # corresponds to the original
        self.data_info = data_info  # meta data of the original whole dataset

        self.bool_array = self.get_bool_array(self.indices)
        self.bool_array_excl = self.get_bool_array(self.indices_excl_overlap)

        self.coverage = len(self.indices)
        self.coverage_excl = len(self.indices_excl_overlap)

        self.rule_base = rule_base  # the previous level of this rule, i.e., the rule from which "self" is obtained

        self.features = self.data_info.features[indices]
        self.target = self.data_info.target[indices]  # target sub-vector covered by this rule
        self.features_excl_overlap = self.data_info.features[indices_excl_overlap]  # feature rows covered by this rule WITHOUT ruleset's cover
        self.target_excl_overlap = self.data_info.target[indices_excl_overlap]  # target covered by this rule WITHOUT ruleset's cover

        self.nrow, self.ncol = self.data_info.features.shape  # local nrow and local ncol
        self.nrow_excl, self.ncol_excl = self.data_info.features_excl_overlap.shape  # local nrow and ncol, excluding ruleset's cover

        self.condition_matrix = condition_matrix
        self.condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        self.prob_excl = self._calc_probs(target=self.target[indices_excl_overlap])
        self.prob = self._calc_probs(target=self.target[indices])
        self.regret_excl = regret(self.nrow_excl, data_info.num_class)
        self.regret = regret(self.nrow, data_info.num_class)

        self.neglog_likelihood_excl = calc_negloglike(p=self.prob_excl, n=self.nrow_excl)

        self.cl_model = self.get_cl_model_indep_data(self.condition_count)

    def _calc_probs(self, target):
        return calc_probs(target, num_class=self.data_info.num_class, smoothed=False)
    
    def update_cl_model_indep_data(self, icol, cut_option):
        if cut_option == LEFT_CUT:
            if self.condition_matrix[LEFT_CUT, icol] == 1:
                return self.cl_model
            else:
                condition_count = np.array(self.condition_matrix)
                condition_count[icol] += 1
                return self.get_cl_model_indep_data(condition_count)
        else:
            if self.condition_matrix[RIGHT_CUT, icol] == 1:
                return self.cl_model
            else:
                condition_count = np.array(self.condition_matrix)
                condition_count[icol] += 1
                return self.get_cl_model_indep_data(condition_count)

    def get_cl_model_indep_data(self, condition_count):
        num_variables = np.count_nonzero(condition_count)
        if num_variables == 0:
            return 0
        else:
            l_num_variables = self.data_info.cl_model["l_number_of_variables"][num_variables]
            l_which_variables = self.data_info.cl_model["l_which_variables"][num_variables]
            l_cuts = self.data_info.cl_model["l_cut"][0][condition_count == 1] + \
                self.data_info.cl_model["l_cut"][1][condition_count == 2]
            return l_num_variables + l_which_variables + l_cuts  # TODO I NEED TO ENCODE THE NUMBER OF RULES?

    def get_bool_array(self, indices):
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True
        return bool_array

    def grow(self):
        candidate_cuts = self.data_info.candidate_cuts
        for icol in range(self.ncol):
            if self.rule_base is None:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
                                          (candidate_cuts[icol] >= np.min(self.features_excl_overlap[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            else:
                candidate_cuts_icol = candidate_cuts[icol]
            for i, cut in enumerate(candidate_cuts_icol):
                excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
                excl_right_bi_array = ~excl_left_bi_array

                excl_left_normalized_gain, cl_model, two_total_cl_left_excl = self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=None)
                excl_right_normalized_gain, cl_model, two_total_cl_right_excl = self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=cl_model)

                incl_left_bi_array = (self.features[:, icol] < cut)
                incl_right_bi_array = ~incl_left_bi_array

    def calculate_incl_gain(self, incl_bi_array, excl_bi_array, icol, cut_option, cl_model):
        excl_coverage, incl_coverage = np.count_nonzero(excl_bi_array), np.count_nonzero(incl_bi_array)
        if excl_coverage == 0:
            return [-np.Inf, cl_model, self.ruleset.elserule_total_cl + cl_model]

        p_excl = self._calc_probs(self.target_excl_overlap[excl_bi_array])
        p_incl = self._calc_probs(self.target[incl_bi_array])

        modelling_groups = self.ruleset.modelling_groups
        both_negloglike = np.zeros(len(modelling_groups), dtype=float)
        for i, modeling_group in enumerate(modelling_groups):
            # Note: both_negloglike[i] represents negloglike(modelling_group \setdiff rule) + negloglike(modelling_Group, rule) # noqa
            both_negloglike[i] = modeling_group.evaluate_rule_approximate(indices=self.indices[incl_bi_array])  # TODO: implement this later in the ModellingGroup class.

        non_overlapping_negloglike = -excl_coverage * np.sum(p_excl * np.log2(p_incl))
        total_negloglike = non_overlapping_negloglike + np.sum(both_negloglike)

        absolute_gain = self.ruleset.total_cl - total_negloglike - regret(incl_coverage, self.data_info.num_class) - cl_model # TODO: implement this later
        normalized_gain = absolute_gain / excl_coverage




    def calculate_excl_gain(self, bi_array, icol, cut_option, cl_model):
        if cl_model is None:
            cl_model = self.update_cl_model_indep_data(icol, cut_option)

        coverage = np.count_nonzero(bi_array)

        if coverage == 0:
            return [-np.Inf, cl_model, self.ruleset.elserule_total_cl + cl_model]

        p = self._calc_probs(self.target_excl_overlap[bi_array])
        negloglike = calc_negloglike(p, coverage)

        else_bool = np.zeros(self.data_info.nrow, dtype=bool)
        else_coverage = np.count_nonzero(else_bool)

        else_bool[self.ruleset.else_indices] = True
        else_bool[self.indices_excl_overlap[bi_array]] = False
        else_p = self._calc_probs(self.data_info.target[else_bool])
        else_negloglike = calc_negloglike(else_p, else_coverage)

        both_negloglike = negloglike + else_negloglike
        both_regret = regret(else_coverage, self.data_info.num_class) + regret(coverage, self.data_info.num_class)

        both_total_cl = both_negloglike + both_regret + cl_model  # "Both" is to emphasize that we ignore the rules already added to the ruleset.

        normalized_gain = (self.ruleset.elserule_total_cl - both_total_cl) / coverage  # TODO: get the elserule_total_cl
        return [normalized_gain, cl_model, both_total_cl]
