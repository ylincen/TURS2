import math
import platform
import sys

import numpy as np
from turs2.utils_calculating_cl import *
from turs2.nml_regret import *
from functools import partial
from utils_readable import *
from turs2.Beam import *

def store_grow_info(excl_bi_array, incl_bi_array, icol, cut, cut_option, excl_mdl_gain, incl_mdl_gain,
                    coverage_excl, coverage_incl):
    return {"excl_bi_array": excl_bi_array, "incl_bi_array": incl_bi_array,
            "icol": icol, "cut": cut, "cut_option": cut_option,
            "excl_mdl_gain": excl_mdl_gain,
            "incl_mdl_gain": incl_mdl_gain,
            "coverage_incl": coverage_incl, "coverage_excl": coverage_excl}

def store_grow_info_rulelist(excl_bi_array, icol, cut, cut_option, excl_normalized_gain):
    return {"excl_bi_array": excl_bi_array, "icol": icol, "cut": cut,
            "cut_option": cut_option, "excl_normalized_gain": excl_normalized_gain}


class Rule:
    def __init__(self, indices, indices_excl_overlap, data_info, rule_base, condition_matrix, ruleset,
                 excl_mdl_gain, incl_mdl_gain, icols_in_order):
        self.ruleset = ruleset
        self.data_info = data_info  # meta data of the original whole dataset
        self.rule_base = rule_base  # the previous level of this rule, i.e., the rule from which "self" is obtained
        self.icols_in_order = icols_in_order

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

        # self.cl_model = self.ruleset.model_encoding.rule_cl_model(self.condition_count)
        self.cl_model = self.ruleset.model_encoding.rule_cl_model_dep(self.condition_matrix, col_orders=icols_in_order)

        if self.rule_base is None:
            self.incl_mdl_gain, self.excl_mdl_gain = -np.Inf, -np.Inf
            self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = -np.Inf, -np.Inf
        else:
            self.incl_mdl_gain, self.excl_mdl_gain = incl_mdl_gain, excl_mdl_gain
            if self.coverage_excl == 0:
                self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = np.nan, np.nan
            else:
                self.incl_gain_per_excl_coverage, self.excl_gain_per_excl_coverage = incl_mdl_gain / self.coverage_excl, excl_mdl_gain / self.coverage_excl

    def new_rule_after_deleting_condition(self, icols, new_ruleset):
        """
        new_ruleset: the ruleset after deleting this rule, for calculating the cl_data aftering adding the new_rule back
        icols: variables (icols) to delete in self
        """
        condition_matrix = np.array(self.condition_matrix)
        condition_count = np.array(self.condition_count)
        for icol in icols:
            if self.condition_count[icol] == 0:
                # print( "The " + str(icol) + "th feature is not in this rule!"  )
                continue
            else:
                condition_matrix[:, icol] = np.nan
                condition_count[icol] = 0

        # initilize the new rule
        new_rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=new_ruleset.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=new_ruleset, excl_mdl_gain=-np.Inf, incl_mdl_gain=-np.Inf,
                        icols_in_order=[ic for ic in self.icols_in_order if ic not in icols])

        # grow the rule by following the condition_matrix
        for icol in new_rule.icols_in_order:
            left_cut, right_cut = condition_matrix[0, icol], condition_matrix[1, icol]

            if not np.isnan(left_cut):
                local_features = self.data_info.features[new_rule.indices, :]
                local_features_excl = self.data_info.features[new_rule.indices_excl_overlap, :]
                incl_bi_array = local_features[:, icol] < left_cut
                excl_bi_array = local_features_excl[:, icol] < left_cut

                grow_info = store_grow_info(excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array, icol=icol,
                                            cut=left_cut, cut_option=LEFT_CUT,
                                            incl_mdl_gain=np.nan, excl_mdl_gain=np.nan,
                                            coverage_excl=np.count_nonzero(excl_bi_array), coverage_incl=np.count_nonzero(incl_bi_array))
                new_rule = new_rule.make_rule_from_grow_info(grow_info=grow_info)

            if not np.isnan(right_cut):
                local_features = self.data_info.features[new_rule.indices, :]
                local_features_excl = self.data_info.features[new_rule.indices_excl_overlap, :]

                incl_bi_array = local_features[:, icol] >= right_cut
                excl_bi_array = local_features_excl[:, icol] >= right_cut

                grow_info = store_grow_info(excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array, icol=icol,
                                            cut=right_cut, cut_option=RIGHT_CUT,
                                            incl_mdl_gain=np.nan, excl_mdl_gain=np.nan,
                                            coverage_excl=np.count_nonzero(excl_bi_array),
                                            coverage_incl=np.count_nonzero(incl_bi_array))
                new_rule = new_rule.make_rule_from_grow_info(grow_info=grow_info)

            new_rule_cl_data_excl = \
                new_rule.ruleset.data_encoding.get_cl_data_excl(ruleset=new_ruleset, rule=new_rule,
                                                                bool=np.ones(new_rule.coverage_excl, dtype=bool))
            new_rule_cl_data_incl = \
                new_rule.ruleset.data_encoding.get_cl_data_incl(ruleset=new_ruleset, rule=new_rule,
                                                                excl_bi_array=np.ones(new_rule.coverage_excl, dtype=bool),
                                                                incl_bi_array=np.ones(new_rule.coverage, dtype=bool))
            sys.exit("Error: below seems wrong! Don't know how to fix this though..")
            new_rule.incl_mdl_gain = new_rule_cl_data_incl / new_rule.coverage_excl
            new_rule.excl_mdl_gain = new_rule_cl_data_excl / new_rule.coverage_excl
        return new_rule

    def _calc_probs(self, target):
        return calc_probs(target, num_class=self.data_info.num_class, smoothed=False)

    def get_bool_array(self, indices):
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True

        return bool_array

    # def grow_rulelist(self):
    #     candidate_cuts = self.data_info.candidate_cuts
    #     excl_best_normalized_gain = -np.Inf
    #     for icol in range(self.ncol):
    #         candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    #                                   (candidate_cuts[icol] >= np.min(self.features_excl_overlap[:, icol]))
    #         candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #
    #         for i, cut in enumerate(candidate_cuts_icol):
    #             excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
    #             excl_right_bi_array = ~excl_left_bi_array
    #
    #             excl_left_normalized_gain, cl_model, two_total_cl_left_excl = self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT)
    #             excl_right_normalized_gain, cl_model, two_total_cl_right_excl = self.calculate_excl_gain(bi_array=excl_right_bi_array, icol=icol, cut_option=LEFT_CUT)
    #
    #             if excl_left_normalized_gain > excl_best_normalized_gain and excl_left_normalized_gain > excl_right_normalized_gain:
    #                 best_excl_grow_info = store_grow_info_rulelist(excl_left_bi_array, icol, cut, LEFT_CUT, excl_left_normalized_gain)
    #                 excl_best_normalized_gain = excl_left_normalized_gain
    #             elif excl_right_normalized_gain > excl_best_normalized_gain and excl_right_normalized_gain > excl_left_normalized_gain:
    #                 best_excl_grow_info = store_grow_info_rulelist(excl_right_bi_array, icol, cut, RIGHT_CUT, excl_right_normalized_gain)
    #                 excl_best_normalized_gain = excl_right_normalized_gain
    #             else:
    #                 pass
    #     excl_grow_res = self.make_rule_from_grow_info_rulelist(best_excl_grow_info)
    #     return excl_grow_res

    def check_split_validity(self, icol, cut):
        indices_left, indices_right = self.indices[self.features[:, icol] < cut], self.indices[self.features[:, icol] >= cut]

        p_rule = self.prob
        p_left = calc_probs(self.data_info.target[indices_left], self.data_info.num_class)
        p_right = calc_probs(self.data_info.target[indices_right], self.data_info.num_class)

        nll_rule = calc_negloglike(p_rule, self.coverage)
        nll_left = calc_negloglike(p_left, len(indices_left))
        nll_right = calc_negloglike(p_right, len(indices_right))

        cl_model_extra = self.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(self.ruleset.model_encoding.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_count > 0)
        cl_model_extra += self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                          self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

        validity = nll_rule + regret(self.coverage, 2) - nll_left - nll_right - regret(len(indices_left), 2) - regret(len(indices_right), 2) - cl_model_extra

        validity_larger_than_zero = (validity > 0)

        return validity_larger_than_zero

    def check_split_validity_excl_with_biarray(self, icol, bi_array_excl):
        indices_left_excl, indices_right_excl = self.indices_excl_overlap[bi_array_excl], self.indices[~bi_array_excl]
        p_rule = self.prob_excl
        p_left = calc_probs(self.data_info.target[indices_left_excl], self.data_info.num_class)
        p_right = calc_probs(self.data_info.target[indices_right_excl], self.data_info.num_class)

        nll_rule = calc_negloglike(p_rule, self.coverage_excl)
        nll_left = calc_negloglike(p_left, len(indices_left_excl))
        nll_right = calc_negloglike(p_right, len(indices_right_excl))

        cl_model_extra = self.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(self.ruleset.model_encoding.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_count > 0)
        cl_model_extra += self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                          self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

        validity = nll_rule + regret(self.coverage_excl, 2) - nll_left - nll_right - regret(len(indices_left_excl), 2) - \
                   regret(len(indices_right_excl), 2) - cl_model_extra
        validity_larger_than_zero = (validity > 0)
        return validity_larger_than_zero

    def check_split_validity_excl(self, icol, cut):
        indices_left_excl, indices_right_excl = self.indices_excl_overlap[self.features_excl_overlap[:, icol] < cut], \
            self.indices_excl_overlap[self.features_excl_overlap[:, icol] >= cut]

        p_rule = self.prob_excl
        p_left = calc_probs(self.data_info.target[indices_left_excl], self.data_info.num_class)
        p_right = calc_probs(self.data_info.target[indices_right_excl], self.data_info.num_class)

        nll_rule = calc_negloglike(p_rule, self.coverage_excl)
        nll_left = calc_negloglike(p_left, len(indices_left_excl))
        nll_right = calc_negloglike(p_right, len(indices_right_excl))

        cl_model_extra = self.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
        cl_model_extra += np.log2(self.ruleset.model_encoding.data_ncol_for_encoding)
        num_vars = np.sum(self.condition_count > 0)
        cl_model_extra += self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                          self.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

        validity = nll_rule + regret(self.coverage_excl, 2) - nll_left - nll_right - regret(len(indices_left_excl), 2) - \
                   regret(len(indices_right_excl), 2) - cl_model_extra

        validity_larger_than_zero = (validity > 0)

        return validity_larger_than_zero

    def validity_check(self, icol, cut):
        if self.data_info.alg_config.validity_check == "no_check":
            check_split_validity = True
        elif self.data_info.alg_config.validity_check == "excl_check":
            check_split_validity = self.check_split_validity_excl(icol, cut)
        elif self.data_info.alg_config.validity_check == "incl_check":
            check_split_validity = self.check_split_validity(icol, cut)  # TODO: this may also cause problems for excl_grow after many rules are already added to the ruleset
        else:
            sys.exit("Error: the if-else statement should not end up here")
        return check_split_validity

    def get_candidate_cuts_icol_given_rule(self, candidate_cuts, icol):
        if self.rule_base is None:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
                                      (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        else:
            candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
                                      (candidate_cuts[icol] > np.min(self.features[:, icol]))
            candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
        return candidate_cuts_icol
    def grow_incl_and_excl_return_beam(self, constraints=None):
        candidate_cuts = self.data_info.candidate_cuts

        if self.data_info.alg_config.beamsearch_positive_gain_only:
            best_incl_mdl_gain, best_excl_mdl_gain = 0, 0
        else:
            best_incl_mdl_gain, best_excl_mdl_gain = -np.Inf, -np.Inf

        invalid_cuts = {}
        best_incl_grow_beam = GrowInfoBeam(width=self.data_info.beam_width)
        best_excl_grow_beam = GrowInfoBeam(width=self.data_info.beam_width)

        for icol in range(self.ncol):
            if constraints is not None and "icols_to_skip" in constraints and \
                    icol in constraints["icols_to_skip"]:
                continue

            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)
            for i, cut in enumerate(candidate_cuts_icol):
                check_split_validity = self.validity_check(icol, cut)

                if not check_split_validity:
                    if icol in invalid_cuts:
                        invalid_cuts[icol].append(cut)
                    else:
                        invalid_cuts[icol] = [cut]
                    continue

                if np.all(self.features[:, icol] < cut) or np.all(self.features[:, icol] >= cut):
                    continue

                best_excl_grow_info, best_incl_grow_info, best_excl_mdl_gain, best_incl_mdl_gain, return_II \
                    = self.search_for_this_cut(icol=icol, option=None, cut=cut,
                                               best_excl_mdl_gain=best_excl_mdl_gain,
                                               best_incl_mdl_gain=best_incl_mdl_gain)

                if best_excl_grow_info is not None:
                    best_excl_grow_beam.update_check_multiple(best_excl_grow_info, best_excl_mdl_gain)
                    best_excl_mdl_gain = best_excl_grow_beam.worst_gain

                    # if self.data_info.alg_config.beamsearch_normalized_gain_must_increase_comparing_rulebase and \
                    #         best_excl_mdl_gain/best_excl_grow_info["coverage_excl"] < self.excl_mdl_gain / self.coverage_excl:
                    #     pass

                if best_incl_grow_info is not None:
                    best_incl_grow_beam.update_check_multiple(best_incl_grow_info, best_incl_mdl_gain)
                    best_incl_mdl_gain = best_incl_grow_beam.worst_gain
                    # if self.data_info.alg_config.beamsearch_normalized_gain_must_increase_comparing_rulebase and \
                    #         best_incl_mdl_gain/best_incl_grow_info["coverage_excl"] < self.incl_mdl_gain / self.coverage_excl:
                    #     pass

        excl_grow_res_beam = [self.make_rule_from_grow_info(excl_grow_info) for excl_grow_info in best_excl_grow_beam.infos]

        if len(best_incl_grow_beam.infos) > 0:
            incl_grow_res_beam = [self.make_rule_from_grow_info(incl_grow_info) for incl_grow_info in best_incl_grow_beam.infos]
        elif self.data_info.alg_config.rerun_on_invalid:
            incl_grow_res_beam = self.rerun_on_invalid(invalid_cuts, constraints, best_incl_grow_beam)
        else:
            incl_grow_res_beam = []
        return [excl_grow_res_beam, incl_grow_res_beam]

    def rerun_on_invalid(self, invalid_cuts, constraints, best_incl_grow_beam):
        assert self.data_info.alg_config.validity_check != "no_check"
        best_incl_mdl_gain = self.incl_mdl_gain
        best_excl_mdl_gain = 0

        if self.data_info.alg_config.rerun_positive_control and self.incl_mdl_gain == -np.inf:
            best_incl_mdl_gain = 0

        for icol in invalid_cuts.keys():
            if constraints is not None and "icols_to_skip" in constraints and \
                    icol in constraints["icols_to_skip"]:
                continue
            for cut in invalid_cuts[icol]:
                if np.all(self.features[:, icol] < cut) or np.all(self.features[:, icol] >= cut):
                    continue
                best_excl_grow_info, best_incl_grow_info, best_excl_mdl_gain, best_incl_mdl_gain, return_II \
                    = self.search_for_this_cut(
                    icol=icol, option=None, cut=cut, best_excl_mdl_gain=best_excl_mdl_gain,
                    best_incl_mdl_gain=best_incl_mdl_gain)
                best_excl_mdl_gain = 0

                if best_incl_grow_info is not None:
                    best_incl_grow_beam.update_check_multiple(best_incl_grow_info, best_incl_mdl_gain)
                    best_incl_mdl_gain = best_incl_grow_beam.worst_gain
                    # if self.data_info.alg_config.beamsearch_normalized_gain_must_increase_comparing_rulebase and \
                    #         best_incl_mdl_gain / best_incl_grow_info[
                    #     "coverage_excl"] < self.incl_mdl_gain / self.coverage_excl:
                    #     pass

        incl_grow_res_beam = [self.make_rule_from_grow_info(incl_grow_info) for incl_grow_info in
                              best_incl_grow_beam.infos]
        return incl_grow_res_beam


    def grow_incl_and_excl(self, constraints=None):
        best_excl_grow_info, best_incl_grow_info = None, None
        candidate_cuts = self.data_info.candidate_cuts
        best_excl_mdl_gain, best_incl_mdl_gain = -np.Inf, -np.Inf

        invalid_cuts = {}

        for icol in range(self.ncol):
            if constraints is not None and "icols_to_skip" in constraints and \
                    icol in constraints["icols_to_skip"]:
                continue

            if self.rule_base is None:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            else:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(self.features[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
            for i, cut in enumerate(candidate_cuts_icol):
                # check_split_validity = self.check_split_validity(icol, cut)  # TODO: this may also cause problems for excl_grow after many rules are already added to the ruleset
                # check_split_validity = self.check_split_validity_excl(icol, cut)
                # if not check_split_validity:
                #     if icol in invalid_cuts:
                #         invalid_cuts[icol].append(cut)
                #     else:
                #         invalid_cuts[icol] = [cut]
                #     continue
                self.check_split_validity_res = True

                # skip (icol, cut) that leads to empty child.
                if np.all(self.features[:, icol] < cut) or np.all(self.features[:, icol] >= cut):
                    continue

                best_excl_grow_info, best_incl_grow_info, best_excl_mdl_gain, best_incl_mdl_gain \
                    = self.search_for_this_cut(
                    icol=icol, option=None, cut=cut, best_excl_mdl_gain=best_excl_mdl_gain,
                    best_incl_mdl_gain=best_incl_mdl_gain)

        # TODO: not clear when "best_excl_grow_info is not None" will be FALSE.
        if best_excl_grow_info is not None:
            excl_grow_res = self.make_rule_from_grow_info(best_excl_grow_info)
        else:
            excl_grow_res = None
        if best_incl_grow_info is not None:
            incl_grow_res = self.make_rule_from_grow_info(best_incl_grow_info)
        else:
            incl_best_normalized_gain = self.incl_normalized_gain

            for icol in invalid_cuts.keys():
                if constraints is not None and "icols_to_skip" in constraints and \
                        icol in constraints["icols_to_skip"]:
                    continue
                for cut in invalid_cuts[icol]:
                    if np.all(self.features[:, icol] < cut) or np.all(self.features[:, icol] >= cut):
                        continue
                    best_excl_grow_info, best_incl_grow_info, excl_best_normalized_gain, incl_best_normalized_gain \
                        = self.search_for_this_cut(
                        icol=icol, option=None, cut=cut, excl_best_normalized_gain=excl_best_normalized_gain,
                        incl_best_normalized_gain=incl_best_normalized_gain,
                        best_excl_grow_info=best_excl_grow_info, best_incl_grow_info=best_incl_grow_info)

            if best_incl_grow_info is not None:
                incl_grow_res = self.make_rule_from_grow_info(best_incl_grow_info)
            else:
                incl_grow_res = None

        return [excl_grow_res, incl_grow_res]

    def search_for_this_cut(self, option, icol, cut, best_excl_mdl_gain, best_incl_mdl_gain):
        if option is None:
            return self.search_for_this_cut_left_and_right(icol, cut, best_excl_mdl_gain, best_incl_mdl_gain)
        elif option == LEFT_CUT:
            return self.search_for_this_cut_left_only(icol, cut, best_excl_mdl_gain,
                                                      best_incl_mdl_gain)
        elif option == RIGHT_CUT:
            return self.search_for_this_cut_right_only(icol, cut, best_excl_mdl_gain,
                                                       best_incl_mdl_gain)
        else:
            pass

    def search_for_this_cut_left_only(self, icol, cut, excl_best_normalized_gain, incl_best_normalized_gain,
                                      best_excl_grow_info, best_incl_grow_info):
        excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
        excl_left_normalized_gain, cl_model, two_total_cl_left_excl = \
            self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT)

        incl_left_bi_array = (self.features[:, icol] < cut)
        incl_left_normalized_gain, cl_model, total_negloglike = \
            self.calculate_incl_gain(incl_bi_array=incl_left_bi_array, excl_bi_array=excl_left_bi_array,
                                     icol=icol, cut_option=LEFT_CUT)

        if excl_left_normalized_gain > excl_best_normalized_gain:
            best_excl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
                                                  excl_left_normalized_gain, incl_left_normalized_gain)
            excl_best_normalized_gain = excl_left_normalized_gain

        if incl_left_normalized_gain > incl_best_normalized_gain:
            best_incl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
                                                  excl_left_normalized_gain, incl_left_normalized_gain)
            incl_best_normalized_gain = incl_left_normalized_gain

        return [best_excl_grow_info, best_incl_grow_info, excl_best_normalized_gain, incl_best_normalized_gain]

    def search_for_this_cut_right_only(self, icol, cut, excl_best_normalized_gain, incl_best_normalized_gain,
                                       best_excl_grow_info, best_incl_grow_info):
        excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
        excl_right_bi_array = ~excl_left_bi_array

        excl_right_normalized_gain, cl_model, two_total_cl_right_excl = \
            self.calculate_excl_gain(bi_array=excl_right_bi_array, icol=icol, cut_option=RIGHT_CUT)

        incl_left_bi_array = (self.features[:, icol] < cut)
        incl_right_bi_array = ~incl_left_bi_array

        incl_right_normalized_gain, cl_model, total_negloglike = \
            self.calculate_incl_gain(incl_bi_array=incl_right_bi_array, excl_bi_array=excl_right_bi_array,
                                     icol=icol, cut_option=RIGHT_CUT)
        if excl_right_normalized_gain > excl_best_normalized_gain:
            best_excl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT,
                                                  excl_right_normalized_gain, incl_right_normalized_gain)
            excl_best_normalized_gain = excl_right_normalized_gain
        if incl_right_normalized_gain > incl_best_normalized_gain:
            best_incl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT,
                                                  excl_right_normalized_gain, incl_right_normalized_gain)
            incl_best_normalized_gain = incl_right_normalized_gain
        return [best_excl_grow_info, best_incl_grow_info, excl_best_normalized_gain, incl_best_normalized_gain]

    def search_for_this_cut_left_and_right(self, icol, cut, best_excl_mdl_gain, best_incl_mdl_gain):
        return_II = False

        new_best_excl_mdl_gain, new_best_incl_mdl_gain = best_excl_mdl_gain, best_incl_mdl_gain

        best_excl_grow_info, best_incl_grow_info = None, None
        excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
        excl_right_bi_array = ~excl_left_bi_array

        excl_left_gain, cl_model, two_total_cl_left_excl = \
            self.calculate_excl_gain(bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT)
        excl_right_gain, cl_model, two_total_cl_right_excl = \
            self.calculate_excl_gain(bi_array=excl_right_bi_array, icol=icol, cut_option=RIGHT_CUT)

        incl_left_bi_array = (self.features[:, icol] < cut)
        incl_right_bi_array = ~incl_left_bi_array

        incl_left_coverage, incl_right_coverage = np.count_nonzero(incl_left_bi_array), np.count_nonzero(incl_right_bi_array)
        excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(excl_right_bi_array)
        if excl_left_coverage == 0 or excl_right_coverage == 0:
            return [best_excl_grow_info, best_incl_grow_info, -np.inf, -np.inf, return_II]

        assert incl_left_coverage > 0
        assert incl_right_coverage > 0

        incl_left_gain, cl_model, total_negloglike = \
            self.calculate_incl_gain(incl_bi_array=incl_left_bi_array, excl_bi_array=excl_left_bi_array,
                                     icol=icol, cut_option=LEFT_CUT)
        incl_right_gain, cl_model, total_negloglike = \
            self.calculate_incl_gain(incl_bi_array=incl_right_bi_array, excl_bi_array=excl_right_bi_array,
                                     icol=icol, cut_option=RIGHT_CUT)

        counter_excl = 0
        if excl_left_gain > best_excl_mdl_gain:
            best_excl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
                                                  excl_left_gain, incl_left_gain,
                                                  coverage_excl=excl_left_coverage,
                                                  coverage_incl=incl_left_coverage)
            new_best_excl_mdl_gain = excl_left_gain
            counter_excl += 1

        if excl_right_gain > best_excl_mdl_gain:
            best_excl_grow_info_ = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT,
                                                  excl_right_gain, incl_right_gain,
                                                  coverage_excl=excl_right_coverage,
                                                  coverage_incl=incl_right_coverage)
            if counter_excl == 0:
                new_best_excl_mdl_gain = excl_right_gain
                best_excl_grow_info = best_excl_grow_info_
            else:
                new_best_excl_mdl_gain = [new_best_excl_mdl_gain, excl_right_gain]
                best_excl_grow_info = [best_excl_grow_info, best_excl_grow_info_]
                return_II = True

        counter_incl = 0
        if incl_left_gain > best_incl_mdl_gain:
            best_incl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
                                                  excl_left_gain, incl_left_gain,
                                                  coverage_excl=excl_left_coverage,
                                                  coverage_incl=incl_left_coverage
                                                  )
            new_best_incl_mdl_gain = incl_left_gain
            counter_incl += 1
        if incl_right_gain > best_incl_mdl_gain:
            best_incl_grow_info_ = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut, RIGHT_CUT,
                                                  excl_right_gain, incl_right_gain,
                                                  coverage_excl=excl_right_coverage,
                                                  coverage_incl=incl_right_coverage
                                                  )
            if counter_incl == 0:
                new_best_incl_mdl_gain = incl_right_gain
                best_incl_grow_info = best_incl_grow_info_
            else:
                new_best_incl_mdl_gain = [new_best_incl_mdl_gain, incl_right_gain]
                best_incl_grow_info = [best_incl_grow_info, best_incl_grow_info_]
                return_II = True

        return [best_excl_grow_info, best_incl_grow_info, new_best_excl_mdl_gain, new_best_incl_mdl_gain, return_II]

    def make_rule_from_grow_info(self, grow_info):
        indices = self.indices[grow_info["incl_bi_array"]]
        indices_excl_overlap = self.indices_excl_overlap[grow_info["excl_bi_array"]]

        condition_matrix = np.array(self.condition_matrix)
        condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
        if grow_info["icol"] in self.icols_in_order:
            new_icols_in_order = self.icols_in_order
        else:
            new_icols_in_order = self.icols_in_order + [grow_info["icol"]]
        rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=self.data_info,
                    rule_base=self, condition_matrix=condition_matrix, ruleset=self.ruleset,
                    excl_mdl_gain=grow_info["excl_mdl_gain"],
                    incl_mdl_gain=grow_info["incl_mdl_gain"],
                    icols_in_order=new_icols_in_order)
        return rule

    # def make_rule_from_grow_info_rulelist(self, grow_info):
    #     indices_excl_overlap = self.indices_excl_overlap[grow_info["excl_bi_array"]]
    #     indices = indices_excl_overlap
    #
    #     condition_matrix = np.array(self.condition_matrix)
    #     condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
    #     rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=self.data_info,
    #                 rule_base=self, condition_matrix=condition_matrix, ruleset=self.ruleset,
    #                 excl_normalized_gain=grow_info["excl_normalized_gain"],
    #                 incl_normalized_gain=grow_info["excl_normalized_gain"])
    #     return rule

    def calculate_excl_gain(self, bi_array, icol, cut_option):
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_data = data_encoding.get_cl_data_excl(self.ruleset, self, bi_array)
        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol, cut_option=cut_option)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model

        return [absolute_gain, cl_model, cl_data]

    def calculate_incl_gain(self, incl_bi_array, excl_bi_array, icol, cut_option):
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_data = data_encoding.get_cl_data_incl(ruleset=self.ruleset, rule=self, excl_bi_array=excl_bi_array, incl_bi_array=incl_bi_array)
        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol, cut_option=cut_option)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model

        return [absolute_gain, cl_model, cl_data]

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
    # candidate_cuts = self.data_info.candidate_cuts
    # excl_best_normalized_gain, incl_best_normalized_gain = -np.Inf, -np.Inf
    # for icol in range(self.ncol):
    # if self.rule_base is None:
    # candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    # (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
    # candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    # else:
    # candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
    # (candidate_cuts[icol] > np.min(self.features[:, icol]))
    # candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    # for i, cut in enumerate(candidate_cuts_icol):
    # excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
    # excl_right_bi_array = ~excl_left_bi_array
    #
    # excl_left_normalized_gain, cl_model, two_total_cl_left_excl = self.calculate_excl_gain(
    # bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=None, for_rule_set=True)
    # excl_right_normalized_gain, cl_model, two_total_cl_right_excl = self.calculate_excl_gain(
    # bi_array=excl_right_bi_array, icol=icol, cut_option=LEFT_CUT, cl_model=cl_model, for_rule_set=True)
    #
    # incl_left_bi_array = (self.features[:, icol] < cut)
    # incl_right_bi_array = ~incl_left_bi_array
    #
    # incl_left_normalized_gain, cl_model, total_negloglike = self.calculate_incl_gain(
    # incl_bi_array=incl_left_bi_array, excl_bi_array=excl_left_bi_array, icol=icol, cut_option=LEFT_CUT,
    # cl_model=cl_model)
    # incl_right_normalized_gain, cl_model, total_negloglike = self.calculate_incl_gain(
    # incl_bi_array=incl_right_bi_array, excl_bi_array=excl_right_bi_array, icol=icol,
    # cut_option=RIGHT_CUT, cl_model=cl_model)
    #
    # if excl_left_normalized_gain > excl_best_normalized_gain and excl_left_normalized_gain >= excl_right_normalized_gain:
    # excl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
    # excl_left_normalized_gain, incl_left_normalized_gain)
    # grow_info_beam_excl.update(excl_grow_info)
    # elif excl_right_normalized_gain > excl_best_normalized_gain and excl_right_normalized_gain > excl_left_normalized_gain:
    # excl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut,
    # RIGHT_CUT, excl_right_normalized_gain,
    # incl_right_normalized_gain)
    # grow_info_beam_excl.update(excl_grow_info)
    # else:
    # pass
    #
    # if incl_left_normalized_gain > incl_best_normalized_gain and incl_left_normalized_gain >= incl_right_normalized_gain:
    # incl_grow_info = store_grow_info(excl_left_bi_array, incl_left_bi_array, icol, cut, LEFT_CUT,
    # excl_left_normalized_gain, incl_left_normalized_gain)
    # grow_info_beam_incl.update(excl_grow_info)
    #
    # elif incl_right_normalized_gain > incl_best_normalized_gain and incl_right_normalized_gain > incl_left_normalized_gain:
    # incl_grow_info = store_grow_info(excl_right_bi_array, incl_right_bi_array, icol, cut,
    # RIGHT_CUT, excl_right_normalized_gain,
    # incl_right_normalized_gain)
    # grow_info_beam_incl.update(excl_grow_info)
    #
    # else:
    # pass
