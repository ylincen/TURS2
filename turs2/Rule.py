import math
import platform
import sys
import numpy as np
from functools import partial

from turs2.utils_calculating_cl import *
from turs2.nml_regret import *
from turs2.utils_readable import *
from turs2.Beam import *
from turs2.RuleGrowConstraint import *

from turs2.constant import *

def store_grow_info(excl_bi_array, incl_bi_array, icol, cut, cut_option, excl_mdl_gain, incl_mdl_gain,
                    coverage_excl, coverage_incl, normalized_gain_excl, normalized_gain_incl, _rule):
    excl_coverage = np.count_nonzero(excl_bi_array)
    incl_coverage = np.count_nonzero(incl_bi_array)
    return locals()

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

    def _calc_probs(self, target):
        return calc_probs(target, num_class=self.data_info.num_class, smoothed=False)

    def get_bool_array(self, indices):
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True

        return bool_array

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

    def update_grow_beam(self, bi_array, excl_bi_array, icol, cut, cut_option, incl_coverage, excl_coverage,
                         grow_info_beam, grow_info_beam_excl, _validity):
        info_theo_scores = self.calculate_mdl_gain(bi_array=bi_array, excl_bi_array=excl_bi_array,
                                                   icol=icol, cut_option=cut_option)
        grow_info = store_grow_info(
            excl_bi_array=excl_bi_array, incl_bi_array=bi_array, icol=icol,
            cut=cut, cut_option=cut_option, incl_mdl_gain=info_theo_scores["absolute_gain"],
            excl_mdl_gain=info_theo_scores["absolute_gain_excl"],
            coverage_excl=excl_coverage, coverage_incl=incl_coverage,
            normalized_gain_excl=info_theo_scores["absolute_gain_excl"] / excl_coverage,
            normalized_gain_incl=info_theo_scores["absolute_gain"] / excl_coverage,
            _rule=self
        )

        excl_cov_percent = grow_info["coverage_excl"] / self.coverage_excl
        incl_cov_percent = grow_info["coverage_incl"] / self.coverage

        if _validity["res_excl"]:
            grow_info_beam_excl.update(grow_info, grow_info["normalized_gain_excl"], excl_cov_percent)

        if _validity["res_incl"]:
            grow_info_beam.update(grow_info, grow_info["normalized_gain_incl"], incl_cov_percent)

    def grow(self, grow_info_beam, grow_info_beam_excl):
        candidate_cuts = self.data_info.candidate_cuts
        for icol in range(self.ncol):
            candidate_cuts_icol = self.get_candidate_cuts_icol_given_rule(candidate_cuts, icol)
            for i, cut in enumerate(candidate_cuts_icol):
                _validity = validity_check(rule=self, icol=icol, cut=cut)

                if self.data_info.not_use_excl_:
                    _validity["res_excl"] = False

                if _validity["res_excl"] == False and _validity["res_incl"] == False:
                    continue

                excl_left_bi_array = (self.features_excl_overlap[:, icol] < cut)
                excl_right_bi_array = ~excl_left_bi_array
                left_bi_array = (self.features[:, icol] < cut)
                right_bi_array = ~left_bi_array

                incl_left_coverage, incl_right_coverage = np.count_nonzero(left_bi_array), np.count_nonzero(
                    right_bi_array)
                excl_left_coverage, excl_right_coverage = np.count_nonzero(excl_left_bi_array), np.count_nonzero(
                    excl_right_bi_array)

                if excl_left_coverage == 0 or excl_right_coverage == 0:
                    continue

                self.update_grow_beam(bi_array=left_bi_array, excl_bi_array=excl_left_bi_array, icol=icol,
                                      cut=cut, cut_option=LEFT_CUT,
                                      incl_coverage=incl_left_coverage, excl_coverage=excl_left_coverage,
                                      grow_info_beam=grow_info_beam, grow_info_beam_excl=grow_info_beam_excl,
                                      _validity=_validity)

                self.update_grow_beam(bi_array=right_bi_array, excl_bi_array=excl_right_bi_array, icol=icol,
                                      cut=cut, cut_option=RIGHT_CUT,
                                      incl_coverage=incl_right_coverage, excl_coverage=excl_right_coverage,
                                      grow_info_beam=grow_info_beam, grow_info_beam_excl=grow_info_beam_excl,
                                      _validity=_validity)


    def calculate_mdl_gain(self, bi_array, excl_bi_array, icol, cut_option):
        data_encoding, model_encoding = self.ruleset.data_encoding, self.ruleset.model_encoding

        cl_model = model_encoding.cl_model_after_growing_rule_on_icol(rule=self, ruleset=self.ruleset, icol=icol,
                                                                      cut_option=cut_option)
        cl_data = data_encoding.get_cl_data_incl(self.ruleset, self, excl_bi_array=excl_bi_array, incl_bi_array=bi_array)
        cl_data_excl = data_encoding.get_cl_data_excl(self.ruleset, self, excl_bi_array)

        absolute_gain = self.ruleset.total_cl - cl_data - cl_model
        absolute_gain_excl = self.ruleset.total_cl - cl_data_excl - cl_model

        return {"cl_model": cl_model, "cl_data": cl_data, "cl_data_excl": cl_data_excl,
                "absolute_gain": absolute_gain, "absolute_gain_excl": absolute_gain_excl}

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
