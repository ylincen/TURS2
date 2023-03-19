import numpy as np
from functools import partial

import nml_regret
from turs2.nml_regret import *
from turs2.utils_calculating_cl import *
from turs2.ModellingGroup import *

class DataEncoding:
    def __init__(self, data_info):
        self.data_info = data_info
        self.num_class = data_info.num_class
        self.calc_probs = partial(calc_probs, num_class=data_info.num_class)

    @staticmethod
    def calc_negloglike(p, n):
        return -n * np.sum(np.log2(p[p != 0]) * p[p != 0])


class NMLencoding(DataEncoding):
    def __init__(self, data_info):
        super().__init__(data_info)
        self.allrules_regret = 0

    def update_ruleset_and_get_cl_data_ruleset_after_adding_rule(self, ruleset, rule):
        ruleset.update_else_rule(rule)
        ruleset.elserule_total_cl = self.get_cl_data_elserule(ruleset)

        allrules_negloglike_except_elserule = ruleset.get_negloglike_all_modelling_groups(rule)
        allrules_regret = np.sum([regret(r.coverage, ruleset.data_info.num_class) for r in ruleset.rules])

        cl_data = ruleset.elserule_total_cl + allrules_negloglike_except_elserule + allrules_regret
        allrules_cl_data = allrules_negloglike_except_elserule + allrules_regret

        self.allrules_regret = allrules_regret

        return [cl_data, allrules_cl_data]

    def get_cl_data_elserule(self, ruleset):
        p = self.calc_probs(self.data_info.target[ruleset.uncovered_indices])
        coverage = len(ruleset.uncovered_indices)
        negloglike_rule = -coverage * np.sum(np.log2(p[p != 0]) * p[p != 0])
        reg = regret(coverage, self.data_info.num_class)
        return negloglike_rule + reg

    def get_cl_data_excl(self, ruleset, rule, bool):
        """
        ruleset: Ruleset object
        rule: a Rule object, so we are calculating the cl_data with ruleset + rule
        bool: a boolean array indicating which indices of the rule's cover is used (use for rule growth)
        """
        p_rule = self.calc_probs(rule.target_excl_overlap[bool])
        coverage_rule = np.count_nonzero(bool)
        negloglike_rule = -coverage_rule * np.sum(np.log2(p_rule[p_rule != 0]) * p_rule[p_rule != 0])

        else_bool = np.array(ruleset.uncovered_bool)
        else_bool[rule.indices_excl_overlap[bool]] = False
        coverage_else = np.count_nonzero(else_bool)
        p_else = calc_probs(self.data_info.target[else_bool], self.data_info.num_class)
        negloglike_else = -coverage_else * np.sum(np.log2(p_else[p_else != 0]) * p_else[p_else != 0])

        regret_else, regret_rule = regret(coverage_else, self.num_class), regret(coverage_rule, self.num_class)

        cl_data = negloglike_else + regret_else + negloglike_rule + regret_rule + ruleset.allrules_cl_data  # TODO: ruleset.allrules_cl_data

        return cl_data

    def get_cl_data_incl(self, ruleset, rule, excl_bi_array, incl_bi_array):
        excl_coverage, incl_coverage = np.count_nonzero(excl_bi_array), np.count_nonzero(incl_bi_array)

        p_excl = self.calc_probs(rule.target_excl_overlap[excl_bi_array])
        p_incl = self.calc_probs(rule.target[incl_bi_array])

        modelling_groups = ruleset.modelling_groups
        both_negloglike = np.zeros(len(modelling_groups),
                                   dtype=float)  # "both" in the name is to emphasize that this is the overlap of both the rule and a modelling_group
        for i, modeling_group in enumerate(modelling_groups):
            # Note: both_negloglike[i] represents negloglike(modelling_group \setdiff rule) + negloglike(modelling_Group \and rule) # noqa
            both_negloglike[i] = modeling_group.evaluate_rule_approximate(indices=rule.indices[incl_bi_array])

        # the non-overlapping part for the rule
        non_overlapping_negloglike = -excl_coverage * np.sum(p_excl[p_incl != 0] * np.log2(p_incl[p_incl != 0]))
        rule_regret = regret(incl_coverage, self.num_class)

        new_else_bool = np.zeros(self.data_info.nrow, dtype=bool)
        new_else_bool[ruleset.uncovered_indices] = True
        new_else_bool[rule.indices_excl_overlap[excl_bi_array]] = False
        new_else_coverage = np.count_nonzero(new_else_bool)
        new_else_p = self.calc_probs(self.data_info.target[new_else_bool])

        new_else_negloglike = self.calc_negloglike(p=new_else_p, n=new_else_coverage)
        new_else_regret = regret(new_else_coverage, self.data_info.num_class)

        cl_data = (new_else_negloglike + new_else_regret) + (np.sum(both_negloglike) + self.allrules_regret) + \
                  (non_overlapping_negloglike + rule_regret)

        return cl_data


class PrequentialEncoding(DataEncoding):
    def get_cl_data_excl(self, ruleset, rule, bool):
        rule_cl_data = calc_prequential(rule.target_excl_overlap[bool], self.num_class)

        else_bool = np.zeros(self.data_info.nrow, dtype=bool)
        else_bool[ruleset.uncovered_indices] = True
        else_bool[rule.indices_excl_overlap[bool]] = False

        else_cl_data = calc_prequential(self.data_info.target[else_bool], self.num_class)

        return rule_cl_data + else_cl_data + ruleset.allrules_cl_data

    def get_cl_data_incl(self, ruleset, rule, bool):
        newrule_bool = np.zeros(ruleset.data_info.nrow, dtype=bool)
        newrule_bool[rule.indices[bool]] = True

        for mg in ruleset.modelling_groups:
            intersection_bool = np.bitwise_and(newrule_bool, mg.cover_bool)
            intersection_count = np.count_nonzero(intersection_bool)
            if intersection_count == 0:
                cl_data_both = mg.cl_data
                # NOT FINISHED


