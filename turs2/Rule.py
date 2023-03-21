import math

import numpy as np
from turs2.utils_calculating_cl import *
from turs2.nml_regret import *
from functools import partial


def store_grow_info(excl_bi_array, incl_bi_array, icol, cut, cut_option, excl_normalized_gain, incl_normalized_gain):
    return {"excl_bi_array": excl_bi_array, "incl_bi_array": incl_bi_array, "icol": icol, "cut": cut,
            "cut_option": cut_option, "excl_normalized_gain": excl_normalized_gain,
            "incl_normalized_gain": incl_normalized_gain}


def store_grow_info_rulelist(excl_bi_array, icol, cut, cut_option, excl_normalized_gain):
    return {"excl_bi_array": excl_bi_array, "icol": icol, "cut": cut,
            "cut_option": cut_option, "excl_normalized_gain": excl_normalized_gain}


class Rule:
    def __init__(self, indices, indices_excl_overlap, data_info, rule_base, condition_matrix, ruleset,
                 excl_normalized_gain, incl_normalized_gain):
        self.ruleset = ruleset
        self.data_info = data_info  # meta data of the original whole dataset
        self.rule_base = rule_base  # the previous level of this rule, i.e., the rule from which "self" is obtained

        self.indices = indices  # corresponds to the original dataset
        self.indices_excl_overlap = indices_excl_overlap  # corresponds to the original
        self.bool_array = self.get_bool_array(self.indices)
        self.bool_array_excl = self.get_bool_array(self.indices_excl_overlap)
        self.coverage = len(self.indices)
        self.coverage_excl = len(self.indices_excl_overlap)
        self.features = self.data_info.features[indices]
        self.target = self.data_info.target[indices]  # target sub-vector covered by this rule
        self.features_excl_overlap = self.data_info.features[indices_excl_overlap]  # feature rows covered by this rule WITHOUT ruleset's cover
        self.target_excl_overlap = self.data_info.target[indices_excl_overlap]  # target covered by this rule WITHOUT ruleset's cover
        self.nrow, self.ncol = len(self.indices), self.data_info.features.shape[1]  # local nrow and local ncol
        self.nrow_excl, self.ncol_excl = len(self.indices_excl_overlap), self.data_info.features.shape[1]  # local nrow and ncol, excluding ruleset's cover

        self.condition_matrix = condition_matrix
        self.condition_count = (~np.isnan(condition_matrix[0])).astype(int) + (~np.isnan(condition_matrix[1])).astype(int)

        self.prob_excl = self._calc_probs(target=self.target_excl_overlap)
        self.prob = self._calc_probs(target=self.target)
        self.regret_excl = regret(self.nrow_excl, data_info.num_class)
        self.regret = regret(self.nrow, data_info.num_class)

        self.neglog_likelihood_excl = calc_negloglike(p=self.prob_excl, n=self.nrow_excl)

        # self.cl_model = self.get_cl_model_indep_data(self.condition_count)
        self.cl_model = self.ruleset.model_encoding.rule_cl_model(self.condition_count)

        if self.rule_base is None:
            self.excl_normalized_gain, self.incl_normalized_gain = -np.Inf, -np.Inf
        else:
            self.excl_normalized_gain = excl_normalized_gain
            self.incl_normalized_gain = incl_normalized_gain

    def new_rule_after_deleting_condition(self, icols, new_ruleset):
        """
        new_ruleset: the ruleset after deleting this rule, for calculating the cl_data aftering adding the new_rule back
        """
        condition_matrix = np.array(self.condition_matrix)
        condition_count = np.array(self.condition_count)
        for icol in icols:
            if self.condition_count[icol] == 0:
                print( "The" + str(icol) + "th feature is not in this rule!"  )
                return -1
            else:
                condition_matrix[:, icol] = np.nan
                condition_count[icol] = 0

        new_rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.ruleset.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=new_ruleset, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf)

        for icol in np.where(condition_count != 0)[0]:
            left_cut, right_cut = condition_matrix[0, icol], condition_matrix[1, icol]

            if not np.isnan(left_cut):
                local_features = self.data_info.X[new_rule.indices, :]
                local_features_excl = self.data_info.X[new_rule.indices_excl_overlap, :]
                incl_bi_array = local_features[:, icol] < left_cut
                excl_bi_array = local_features_excl[:, icol] < left_cut

                grow_info = store_grow_info(excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array, icol=icol,
                                            cut=left_cut, cut_option=LEFT_CUT,
                                            incl_normalized_gain=np.nan, excl_normalized_gain=np.nan)
                new_rule = new_rule.make_rule_from_grow_info(grow_info=grow_info)

            if not np.isnan(right_cut):
                local_features = self.data_info.features[new_rule.indices, :]
                local_features_excl = self.data_info.features[new_rule.indices_excl_overlap, :]

                incl_bi_array = local_features[:, icol] >= right_cut
                excl_bi_array = local_features_excl[:, icol] >= right_cut

                grow_info = store_grow_info(excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array, icol=icol,
                                            cut=right_cut, cut_option=RIGHT_CUT,
                                            incl_normalized_gain=np.nan, excl_normalized_gain=np.nan)
                new_rule = new_rule.make_rule_from_grow_info(grow_info=grow_info)

            new_rule_cl_data_excl = \
                self.ruleset.data_encoding.get_cl_data_excl(ruleset=new_ruleset, rule=new_rule,
                                                            bool=np.ones(new_rule.coverage_excl, dtype=bool))
            new_rule_cl_data_incl = \
                self.ruleset.data_encoding.get_cl_data_incl(ruleset=new_ruleset, rule=new_rule,
                                                            excl_bi_array=np.ones(new_rule.coverage_excl, dtype=bool),
                                                            incl_bi_array=np.ones(new_rule.coverage, dtype=bool))

            new_rule.incl_normalized_gain = new_rule_cl_data_incl / new_rule.coverage_excl
            new_rule.excl_normalized_gain = new_rule_cl_data_excl / new_rule.coverage_excl
        return new_rule

    def _calc_probs(self, target):
        return calc_probs(target, num_class=self.data_info.num_class, smoothed=False)
    
    # def update_cl_model_indep_data(self, icol, cut_option):
    #     if cut_option == LEFT_CUT:
    #         if self.condition_matrix[LEFT_CUT, icol] == 1:
    #             return self.cl_model
    #         else:
    #             condition_count = np.array(self.condition_count)
    #             condition_count[icol] += 1
    #             return self.get_cl_model_indep_data(condition_count, update_cl_model_for_debug=False)
    #     else:
    #         if self.condition_matrix[RIGHT_CUT, icol] == 1:
    #             return self.cl_model
    #         else:
    #             condition_count = np.array(self.condition_count)
    #             condition_count[icol] += 1
    #             return self.get_cl_model_indep_data(condition_count, update_cl_model_for_debug=False)
    #
    # def get_cl_model_indep_data(self, condition_count, update_cl_model_for_debug=True):
    #     num_variables = np.count_nonzero(condition_count)
    #     if num_variables == 0:
    #         return 0
    #     else:
    #         l_num_variables = self.data_info.cl_model["l_number_of_variables"][num_variables]
    #
    #         l_which_variables = self.data_info.cl_model["l_which_variables"][num_variables]
    #         l_cuts = np.sum(self.data_info.cl_model["l_cut"][0][condition_count == 1]) + \
    #             np.sum(self.data_info.cl_model["l_cut"][1][condition_count == 2])
    #         if update_cl_model_for_debug:
    #             self.cl_model_for_debug = {"l_num_variables": l_num_variables,
    #                                        "l_which_variables": l_which_variables,
    #                                        "l_cuts": l_cuts}
    #         return l_num_variables + l_which_variables + l_cuts

    def get_bool_array(self, indices):
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True

        return bool_array

    def grow_rulelist(self):
        candidate_cuts = self.data_info.candidate_cuts
        excl_best_normalized_gain = -np.Inf
        for icol in range(self.ncol):
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
                                      (candidate_cuts[icol] >= np.min(self.features_excl_overlap[:, icol]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]

            for i, cut in enumerate(candidate_cuts_icol):
                excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
                excl_right_bi_array = ~excl_left_bi_array

                excl_left_normalized_gain, cl_model, two_total_cl_left_excl = self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT)
                excl_right_normalized_gain, cl_model, two_total_cl_right_excl = self.calculate_excl_gain(bi_array=excl_right_bi_array, icol=icol, cut_option=LEFT_CUT)

                if excl_left_normalized_gain > excl_best_normalized_gain and excl_left_normalized_gain > excl_right_normalized_gain:
                    best_excl_grow_info = store_grow_info_rulelist(excl_left_bi_array, icol, cut, LEFT_CUT, excl_left_normalized_gain)
                    excl_best_normalized_gain = excl_left_normalized_gain
                elif excl_right_normalized_gain > excl_best_normalized_gain and excl_right_normalized_gain > excl_left_normalized_gain:
                    best_excl_grow_info = store_grow_info_rulelist(excl_right_bi_array, icol, cut, RIGHT_CUT, excl_right_normalized_gain)
                    excl_best_normalized_gain = excl_right_normalized_gain
                else:
                    pass
        excl_grow_res = self.make_rule_from_grow_info_rulelist(best_excl_grow_info)
        return excl_grow_res

    def grow_incl_and_excl(self):
        best_excl_grow_info, best_incl_grow_info = None, None
        candidate_cuts = self.data_info.candidate_cuts
        excl_best_normalized_gain, incl_best_normalized_gain = -np.Inf, -np.Inf
        for icol in range(self.ncol):
            if self.rule_base is None:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            else:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(self.features[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            for i, cut in enumerate(candidate_cuts_icol):
                # if icol == 8 and abs(cut - 14.166667405592676) < 0.001 and len(self.ruleset.rules)==4:
                #     print("here")
                excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
                excl_right_bi_array = ~excl_left_bi_array
                # if len(self.indices_excl_overlap[excl_left_bi_array]) == 1 or len(self.indices_excl_overlap[excl_right_bi_array]) == 1:
                #     if self.indices_excl_overlap[excl_left_bi_array][0] == 2498 or self.indices_excl_overlap[excl_right_bi_array][0] == 2498:
                #         print("here")

                excl_left_normalized_gain, cl_model, two_total_cl_left_excl = \
                    self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT)
                excl_right_normalized_gain, cl_model, two_total_cl_right_excl = \
                    self.calculate_excl_gain(bi_array=excl_right_bi_array, icol=icol, cut_option=LEFT_CUT)

                incl_left_bi_array = (self.features[:, icol] < cut)
                incl_right_bi_array = ~incl_left_bi_array

                # if np.array_equal(np.where(incl_right_bi_array)[0], np.array([248, 1256, 1738, 3982])) or np.array_equal(np.where(incl_left_bi_array)[0], np.array([ 248, 1256, 1738, 3982])):
                #     print("debug")
                # if np.count_nonzero(incl_right_bi_array) == 4 or np.count_nonzero(incl_left_bi_array) == 4:
                #     if icol > 140:
                #         print("debug")

                incl_left_normalized_gain, cl_model, total_negloglike = \
                    self.calculate_incl_gain(incl_bi_array=incl_left_bi_array, excl_bi_array=excl_left_bi_array,
                                             icol=icol, cut_option=LEFT_CUT)
                incl_right_normalized_gain, cl_model, total_negloglike = \
                    self.calculate_incl_gain(incl_bi_array=incl_right_bi_array, excl_bi_array=excl_right_bi_array,
                                             icol=icol, cut_option=RIGHT_CUT)

                if excl_left_normalized_gain > excl_best_normalized_gain and excl_left_normalized_gain >= excl_right_normalized_gain:
                    best_excl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT, excl_left_normalized_gain, incl_left_normalized_gain)
                    excl_best_normalized_gain = excl_left_normalized_gain
                elif excl_right_normalized_gain > excl_best_normalized_gain and excl_right_normalized_gain > excl_left_normalized_gain:
                    best_excl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT, excl_right_normalized_gain, incl_right_normalized_gain)
                    excl_best_normalized_gain = excl_right_normalized_gain
                else:
                    pass

                if incl_left_normalized_gain > incl_best_normalized_gain and incl_left_normalized_gain >= incl_right_normalized_gain:
                    best_incl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT, excl_left_normalized_gain, incl_left_normalized_gain)
                    incl_best_normalized_gain = incl_left_normalized_gain
                elif incl_right_normalized_gain > incl_best_normalized_gain and incl_right_normalized_gain > incl_left_normalized_gain:
                    best_incl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT, excl_right_normalized_gain, incl_right_normalized_gain)
                    incl_best_normalized_gain = incl_right_normalized_gain
                else:
                    pass
        if best_excl_grow_info is not None:
            excl_grow_res = self.make_rule_from_grow_info(best_excl_grow_info)
        else:
            excl_grow_res = None
        if best_incl_grow_info is not None:
            incl_grow_res = self.make_rule_from_grow_info(best_incl_grow_info)
        else:
            incl_grow_res = None

        return [excl_grow_res, incl_grow_res]

    def make_rule_from_grow_info(self, grow_info):
        indices = self.indices[grow_info["incl_bi_array"]]
        indices_excl_overlap = self.indices_excl_overlap[grow_info["excl_bi_array"]]

        condition_matrix = np.array(self.condition_matrix)
        condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
        rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=self.data_info,
                    rule_base=self, condition_matrix=condition_matrix, ruleset=self.ruleset,
                    excl_normalized_gain=grow_info["excl_normalized_gain"],
                    incl_normalized_gain=grow_info["incl_normalized_gain"])
        return rule

    def make_rule_from_grow_info_rulelist(self, grow_info):
        indices_excl_overlap = self.indices_excl_overlap[grow_info["excl_bi_array"]]
        indices = indices_excl_overlap

        condition_matrix = np.array(self.condition_matrix)
        condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
        rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=self.data_info,
                    rule_base=self, condition_matrix=condition_matrix, ruleset=self.ruleset,
                    excl_normalized_gain=grow_info["excl_normalized_gain"],
                    incl_normalized_gain=grow_info["excl_normalized_gain"])
        return rule

    def calculate_excl_gain(self, bi_array, icol, cut_option):
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_data = data_encoding.get_cl_data_excl(self.ruleset, self, bi_array)
        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol, cut_option=cut_option)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model
        if np.count_nonzero(bi_array) == 0:
            normalized_gain = -np.Inf
        else:
            normalized_gain = absolute_gain / np.count_nonzero(bi_array)

        return [normalized_gain, cl_model, cl_data]

    def calculate_incl_gain(self, incl_bi_array, excl_bi_array, icol, cut_option):
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_data = data_encoding.get_cl_data_incl(ruleset=self.ruleset, rule=self, excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array)
        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol, cut_option=cut_option)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model
        if np.count_nonzero(excl_bi_array) == 0:
            normalized_gain = -np.Inf
        else:
            normalized_gain = absolute_gain / np.count_nonzero(excl_bi_array)   # use excl_bi_array so the normalized gain is some sort of "learning" rate;

        return [normalized_gain, cl_model, cl_data]


    # def calculate_incl_gain(self, incl_bi_array, excl_bi_array, icol, cut_option, cl_model=None):
    #     excl_coverage, incl_coverage = np.count_nonzero(excl_bi_array), np.count_nonzero(incl_bi_array)
    #
    #     if excl_coverage == 0:
    #         return [-np.Inf, cl_model, -np.Inf]
    #
    #     if cl_model is None:
    #         cl_model = self.update_cl_model_indep_data(icol, cut_option)
    #
    #     p_excl = self._calc_probs(self.target_excl_overlap[excl_bi_array])
    #     p_incl = self._calc_probs(self.target[incl_bi_array])
    #
    #     modelling_groups = self.ruleset.modelling_groups
    #     both_negloglike = np.zeros(len(modelling_groups), dtype=float)  # "both" in the name is to emphasize that this is the overlap of both the rule and a modelling_group
    #     for i, modeling_group in enumerate(modelling_groups):
    #         # Note: both_negloglike[i] represents negloglike(modelling_group \setdiff rule) + negloglike(modelling_Group, rule) # noqa
    #         both_negloglike[i] = modeling_group.evaluate_rule_approximate(indices=self.indices[incl_bi_array],
    #                                                                       rule_prob_incl=p_incl)
    #
    #     non_overlapping_negloglike = -excl_coverage * np.sum(p_excl[p_incl != 0] * np.log2(p_incl[p_incl != 0]))
    #
    #     new_else_bool = np.zeros(self.data_info.nrow, dtype=bool)
    #     new_else_bool[self.ruleset.uncovered_indices] = True
    #     new_else_bool[self.indices_excl_overlap[excl_bi_array]] = False
    #     new_else_coverage = np.count_nonzero(new_else_bool)
    #     new_else_p = calc_probs(self.data_info.target[new_else_bool], self.data_info.num_class)
    #
    #     else_negloglike = calc_negloglike(p=new_else_p, n=new_else_coverage)
    #     new_else_regret = regret(new_else_coverage, self.data_info.num_class)
    #
    #     total_negloglike = else_negloglike + non_overlapping_negloglike + np.sum(both_negloglike)
    #
    #     cl_permutations_of_rules_after_adding = math.lgamma(len(self.ruleset.rules) + 2) / np.log(2)
    #     cl_model_after_adding = self.ruleset.allrules_cl_model + cl_model - cl_permutations_of_rules_after_adding + \
    #         self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules) + 1]
    #
    #     cl_extra_cost_random_design = np.log2(self.ruleset.else_rule_coverage)
    #
    #     total_cl_after_growing = (
    #             total_negloglike +
    #             (
    #                 new_else_regret + regret(incl_coverage, self.data_info.num_class) + self.ruleset.allrules_regret
    #             ) +
    #             cl_model_after_adding + (cl_extra_cost_random_design + self.ruleset.cl_cost_random_design)
    #     )
    #
    #     absolute_gain = self.ruleset.total_cl - total_cl_after_growing
    #     normalized_gain = absolute_gain / excl_coverage
    #
    #     return [normalized_gain, cl_model, total_negloglike]

    # def calculate_excl_gain(self, bi_array, icol, cut_option, cl_model, for_rule_set):
    #     """
    #     for_rule_set: Boolean. Whether it is for building a rule list, or it is for the search of a rule set.
    #     """
    #     if cl_model is None:
    #         cl_model = self.update_cl_model_indep_data(icol, cut_option)
    #
    #     coverage = np.count_nonzero(bi_array)
    #
    #     if coverage == 0:
    #         return [-np.Inf, cl_model, self.ruleset.elserule_total_cl + cl_model]
    #
    #     p = self._calc_probs(self.target_excl_overlap[bi_array])
    #     negloglike = calc_negloglike(p, coverage)
    #
    #     else_bool = np.zeros(self.data_info.nrow, dtype=bool)
    #     else_bool[self.ruleset.uncovered_indices] = True
    #     else_bool[self.indices_excl_overlap[bi_array]] = False
    #     else_coverage = np.count_nonzero(else_bool)
    #
    #     else_p = self._calc_probs(self.data_info.target[else_bool])
    #     else_negloglike = calc_negloglike(else_p, else_coverage)
    #
    #     both_negloglike = negloglike + else_negloglike
    #     both_regret = regret(else_coverage, self.data_info.num_class) + regret(coverage, self.data_info.num_class)
    #
    #     both_total_cl = both_negloglike + both_regret + cl_model  # "Both" is to emphasize that we ignore the rules already added to the ruleset.
    #
    #     if for_rule_set:
    #         cl_extra_cost_number_of_rules = self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules) + 1] - \
    #                                         self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules)]
    #         cl_permutations_of_rules_current = math.lgamma(len(self.ruleset.rules) + 1) / np.log(2)   # log factorial
    #         cl_permutations_of_rules_candidate = math.lgamma(len(self.ruleset.rules) + 2) / np.log(2)
    #         cl_extra_cost_random_design = np.log2(self.ruleset.else_rule_coverage)
    #
    #         normalized_gain = (self.ruleset.elserule_total_cl - cl_permutations_of_rules_current - both_total_cl -
    #                            cl_extra_cost_number_of_rules + cl_permutations_of_rules_candidate - cl_extra_cost_random_design) / coverage
    #     else:
    #         cl_extra_cost_number_of_rules = self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules) + 1] - \
    #                                         self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules)]
    #         cl_extra_cost_random_design = np.log2(self.ruleset.else_rule_coverage)
    #         normalized_gain = (self.ruleset.elserule_total_cl - both_total_cl -
    #                            cl_extra_cost_number_of_rules - cl_extra_cost_random_design) / coverage
    #
    #     return [normalized_gain, cl_model, both_total_cl]

    # def calculate_excl_gain_prequential(self, bi_array, icol, cut_option, cl_model, for_rule_set):
    #     if cl_model is None:
    #             cl_model = self.update_cl_model_indep_data(icol, cut_option)
    #
    #     coverage = np.count_nonzero(bi_array)
    #
    #     if coverage == 0:
    #         return [-np.Inf, cl_model, self.ruleset.elserule_total_cl + cl_model]
    #
    #     negloglike = calc_prequential(self.target_excl_overlap[bi_array], self.data_info.num_class)
    #
    #     else_bool = np.zeros(self.data_info.nrow, dtype=bool)
    #     else_bool[self.ruleset.uncovered_indices] = True
    #     else_bool[self.indices_excl_overlap[bi_array]] = False
    #
    #     else_negloglike = calc_prequential(self.data_info.target[else_bool], self.data_info.num_class)
    #
    #     both_negloglike = negloglike + else_negloglike
    #
    #     both_total_cl = both_negloglike + cl_model
    #
    #     if for_rule_set:
    #         cl_extra_cost_number_of_rules = self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules) + 1] - \
    #                                         self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules)]
    #         cl_permutations_of_rules_current = math.lgamma(len(self.ruleset.rules) + 1) / np.log(2)   # log factorial
    #         cl_permutations_of_rules_candidate = math.lgamma(len(self.ruleset.rules) + 2) / np.log(2)
    #         normalized_gain = (self.ruleset.elserule_total_cl - cl_permutations_of_rules_current - both_total_cl -
    #                            cl_extra_cost_number_of_rules + cl_permutations_of_rules_candidate) / coverage
    #     else:
    #         cl_extra_cost_number_of_rules = self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules) + 1] - \
    #                                         self.data_info.cl_model["l_number_of_rules"][len(self.ruleset.rules)]
    #         normalized_gain = (self.ruleset.elserule_total_cl - both_total_cl -
    #                            cl_extra_cost_number_of_rules) / coverage
    #
    #     return [normalized_gain, cl_model, both_total_cl]

    def _print(self):
        feature_names = self.ruleset.data_info.feature_names
        readable = ""
        which_variables = np.where(self.condition_count != 0)[0]
        for v in which_variables:
            cut = self.condition_matrix[:, v][::-1]
            icol_name = feature_names[v]
            readable += "X" + str(v) + "-" + str(icol_name) + " in " + str(cut) + ";   "

        readable += "Prob: " + str(self.prob) + ", Coverage: " + str(self.coverage)
        return readable

    # def grow_incl_excl_beam(self, grow_info_beam_excl, grow_info_beam_incl):
    #     candidate_cuts = self.data_info.candidate_cuts
    #     excl_best_normalized_gain, incl_best_normalized_gain = -np.Inf, -np.Inf
    #     for icol in range(self.ncol):
    #         if self.rule_base is None:
    #             candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    #                                       (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
    #             candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #         else:
    #             candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
    #                                       (candidate_cuts[icol] > np.min(self.features[:, icol]))
    #             candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #         for i, cut in enumerate(candidate_cuts_icol):
    #             excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
    #             excl_right_bi_array = ~excl_left_bi_array
    #
    #             excl_left_normalized_gain, cl_model, two_total_cl_left_excl = self.calculate_excl_gain(
    #                 bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=None, for_rule_set=True)
    #             excl_right_normalized_gain, cl_model, two_total_cl_right_excl = self.calculate_excl_gain(
    #                 bi_array=excl_right_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=cl_model, for_rule_set=True)
    #
    #             incl_left_bi_array = (self.features[:, icol] < cut)
    #             incl_right_bi_array = ~incl_left_bi_array
    #
    #             incl_left_normalized_gain, cl_model, total_negloglike = self.calculate_incl_gain(
    #                 incl_bi_array=incl_left_bi_array, excl_bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT,
    #                 cl_model=cl_model)
    #             incl_right_normalized_gain, cl_model, total_negloglike = self.calculate_incl_gain(
    #                 incl_bi_array=incl_right_bi_array, excl_bi_array=excl_right_bi_array, icol=icol,
    #                 cut_option=RIGHT_CUT, cl_model=cl_model)
    #
    #             if excl_left_normalized_gain > excl_best_normalized_gain and excl_left_normalized_gain >= excl_right_normalized_gain:
    #                 excl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
    #                                                       excl_left_normalized_gain, incl_left_normalized_gain)
    #                 grow_info_beam_excl.update(excl_grow_info)
    #             elif excl_right_normalized_gain > excl_best_normalized_gain and excl_right_normalized_gain > excl_left_normalized_gain:
    #                 excl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut,
    #                                                       RIGHT_CUT, excl_right_normalized_gain,
    #                                                       incl_right_normalized_gain)
    #                 grow_info_beam_excl.update(excl_grow_info)
    #             else:
    #                 pass
    #
    #             if incl_left_normalized_gain > incl_best_normalized_gain and incl_left_normalized_gain >= incl_right_normalized_gain:
    #                 incl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
    #                                                       excl_left_normalized_gain, incl_left_normalized_gain)
    #                 grow_info_beam_incl.update(excl_grow_info)
    #
    #             elif incl_right_normalized_gain > incl_best_normalized_gain and incl_right_normalized_gain > incl_left_normalized_gain:
    #                 incl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut,
    #                                                       RIGHT_CUT, excl_right_normalized_gain,
    #                                                       incl_right_normalized_gain)
    #                 grow_info_beam_incl.update(excl_grow_info)
    #
    #             else:
    #                 pass