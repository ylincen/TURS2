import numpy as np
import pickle
import os
from datetime import datetime
import platform

from sklearn.metrics import roc_auc_score

from turs2.Rule import *
from turs2.Beam import *
from turs2.ModellingGroup import *
from turs2.DataInfo import *
from turs2.utils_readable import *
from turs2.utils_predict import *


def make_rule_from_grow_info(grow_info):
    rule = grow_info["_rule"]
    indices = rule.indices[grow_info["incl_bi_array"]]
    indices_excl_overlap = rule.indices_excl_overlap[grow_info["excl_bi_array"]]

    condition_matrix = np.array(rule.condition_matrix)
    condition_matrix[grow_info["cut_option"], grow_info["icol"]] = grow_info["cut"]
    if grow_info["icol"] in rule.icols_in_order:
        new_icols_in_order = rule.icols_in_order
    else:
        new_icols_in_order = rule.icols_in_order + [grow_info["icol"]]
    rule = Rule(indices=indices, indices_excl_overlap=indices_excl_overlap, data_info=rule.data_info,
                rule_base=rule, condition_matrix=condition_matrix, ruleset=rule.ruleset,
                excl_mdl_gain=grow_info["excl_mdl_gain"],
                incl_mdl_gain=grow_info["incl_mdl_gain"],
                icols_in_order=new_icols_in_order)
    return rule

def extract_rules_from_beams(beams):
    # beams: a list of beam
    rules = []
    coverage_list = []
    for beam in beams:
        for info in beam.infos:
            if info["coverage_incl"] in coverage_list:
                index_equal = coverage_list.index(info["coverage_incl"])
                if np.all(rules[index_equal].indices == info["_rule"].indices[info["incl_bi_array"]]):
                    continue

            r = make_rule_from_grow_info(grow_info=info)
            rules.append(r)
            coverage_list.append(r.coverage)
    return rules


class Ruleset:
    def __init__(self, data_info, data_encoding, model_encoding, constraints=None):
        self.log_folder_name = None

        self.data_info = data_info
        self.model_encoding = model_encoding
        self.data_encoding = data_encoding

        self.rules = []

        self.uncovered_indices = np.arange(data_info.nrow)
        self.uncovered_bool = np.ones(self.data_info.nrow, dtype=bool)
        self.else_rule_p = calc_probs(target=data_info.target, num_class=data_info.num_class)
        self.else_rule_coverage = self.data_info.nrow
        self.elserule_total_cl = self.data_encoding.get_cl_data_elserule(ruleset=self)

        self.negloglike = -np.sum(data_info.nrow * np.log2(self.else_rule_p[self.else_rule_p != 0]) * self.else_rule_p[self.else_rule_p != 0])
        self.else_rule_negloglike = self.negloglike

        self.cl_data = self.elserule_total_cl
        self.cl_model = 0  # cl model for the whole rule set (including the number of rules)
        self.allrules_cl_model = 0  # cl model for all rules, summed up
        self.total_cl = self.cl_model + self.cl_data  # total cl with 0 rules

        self.modelling_groups = []

        self.allrules_cl_data = 0

        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = constraints

    def add_rule(self, rule):
        self.rules.append(rule)
        self.cl_data, self.allrules_cl_data = \
            self.data_encoding.update_ruleset_and_get_cl_data_ruleset_after_adding_rule(ruleset=self, rule=rule)
        self.cl_model = \
            self.model_encoding.cl_model_after_growing_rule_on_icol(rule=rule, ruleset=self, icol=None, cut_option=None)

        self.total_cl = self.cl_data + self.cl_model
        self.allrules_cl_model += rule.cl_model

    def update_else_rule(self, rule):
        self.uncovered_bool = np.bitwise_and(self.uncovered_bool, ~rule.bool_array)
        self.uncovered_indices = np.where(self.uncovered_bool)[0]
        self.else_rule_coverage = len(self.uncovered_indices)
        self.else_rule_p = calc_probs(self.data_info.target[self.uncovered_indices], self.data_info.num_class)
        self.else_rule_negloglike = calc_negloglike(self.else_rule_p, self.else_rule_coverage)

    def get_negloglike_all_modelling_groups(self, rule):
        all_mg_neglolglike = []
        if len(self.modelling_groups) == 0:
            mg = ModellingGroup(data_info=self.data_info, bool_cover=rule.bool_array,
                                bool_use_for_model=rule.bool_array,
                                rules_involved=[0], prob_model=rule.prob,
                                prob_cover=rule.prob)
            all_mg_neglolglike.append(mg.negloglike)
            self.modelling_groups.append(mg)
        else:
            num_mgs = len(self.modelling_groups)
            for jj in range(num_mgs):
                m = self.modelling_groups[jj]
                evaluate_res = m.evaluate_rule(rule, update_rule_index=len(self.rules) - 1)
                all_mg_neglolglike.append(evaluate_res[0])
                if evaluate_res[1] is not None:
                    self.modelling_groups.append(evaluate_res[1])

            mg = ModellingGroup(data_info=self.data_info,
                                bool_cover=rule.bool_array_excl,
                                bool_use_for_model=rule.bool_array,
                                rules_involved=[len(self.rules) - 1], prob_model=rule.prob,
                                prob_cover=rule.prob_excl)
            all_mg_neglolglike.append(mg.negloglike)
            self.modelling_groups.append(mg)
        return np.sum(all_mg_neglolglike)

    def fit(self, max_iter=1000, printing=True):
        total_cl = [self.total_cl]
        if self.data_info.log_learning_process:
            log_folder_name = self.data_info.log_folder_name

        log_info_ruleset = ""
        for iter in range(max_iter):
            if printing:
                print("iteration ", iter)
            rule_to_add = self.search_next_rule(k_consecutively=5)
            if printing:
                print(print_(rule_to_add))

            add_to_ruleset = rule_to_add.incl_gain_per_excl_coverage > 0

            if add_to_ruleset:
                self.add_rule(rule_to_add)
                total_cl.append(self.total_cl)
                if self.data_info.log_learning_process:
                    log_info_ruleset += "Add rule: " + print_(rule_to_add) + "\n\n"
                    log_info_ruleset += "with grow process: "
                    r = rule_to_add.rule_base
                    while r is not None:
                        log_info_ruleset += print_(r) + "\n"
                        r = r.rule_base
            else:
                break
        if self.data_info.log_learning_process:
            system_name = platform.system()
            if system_name == "Windows":
                with open(log_folder_name + "\\ruleset.txt", "w") as flog_ruleset:
                    flog_ruleset.write(log_info_ruleset)
            else:
                with open(log_folder_name + "/ruleset.txt", "w") as flog_ruleset:
                    flog_ruleset.write(log_info_ruleset)
        return total_cl

    @staticmethod
    def calculate_stop_condition_element(incl_beam, excl_beam, previous_best_gain, previous_best_excl_gain):
        # Condition 3 is actually redundant, as we only add a rule to the rule set if it has positive incl_gain
        condition1 = len(incl_beam.gains) > 0 and np.max(incl_beam.gains) < previous_best_gain
        condition2 = len(excl_beam.gains) > 0 and np.max(excl_beam.gains) < previous_best_excl_gain
        condition3 = len(excl_beam.gains) > 0 and len(incl_beam.gains) > 0 and np.max(incl_beam.gains) < 0 and np.max(excl_beam.gains) < 0
        return (condition1 and condition2) or condition3

    def combine_beams(self, incl_beam_list, excl_beam_list):
        infos_incl, infos_excl = [], []
        coverages_incl, coverages_excl = [], []

        # for incl
        for incl_beam in incl_beam_list:
            infos_incl.extend([info for info in incl_beam.infos.values() if info is not None])
            coverages_incl.extend([info["coverage_incl"] for info in incl_beam.infos.values() if info is not None])
        argsorted_coverages_incl = np.argsort(coverages_incl)
        groups_coverages_incl = np.array_split([infos_incl[i] for i in argsorted_coverages_incl], self.data_info.beam_width)

        final_info_incl = []
        for group in groups_coverages_incl:
            if len(group) == 0:
                continue
            final_info_incl.append(group[np.argmax([info["normalized_gain_incl"] for info in group])])

        # for excl
        for excl_beam in excl_beam_list:
            infos_excl.extend([info for info in excl_beam.infos.values() if info is not None])
            coverages_excl.extend([info["coverage_excl"] for info in excl_beam.infos.values() if info is not None])
        argsorted_coverages_excl = np.argsort(coverages_excl)
        groups_coverages_excl = np.array_split([infos_excl[i] for i in argsorted_coverages_excl], self.data_info.beam_width)
        final_info_excl = []
        for group in groups_coverages_excl:
            if len(group) == 0:
                continue
            final_info_excl.append(group[np.argmax([info["normalized_gain_excl"] for info in group])])
        return final_info_incl, final_info_excl


    def search_next_rule(self, k_consecutively, rule_given=None, constraints=None):
        if rule_given is None:
            rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=self, excl_mdl_gain=-np.Inf, incl_mdl_gain=-np.Inf, icols_in_order=[])
        else:
            rule = rule_given

        rules_for_next_iter = [rule]
        rules_candidates = [rule]

        previous_best_gain, previous_best_excl_gain = -np.Inf, -np.Inf
        counter_worse_best_gain = 0
        for i in range(self.data_info.max_grow_iter):
            excl_beam_list, incl_beam_list = [], []
            for rule in rules_for_next_iter:
                excl_beam = DiverseCovBeam(width=self.data_info.beam_width)
                incl_beam = DiverseCovBeam(width=self.data_info.beam_width)
                rule.grow(grow_info_beam=incl_beam, grow_info_beam_excl=excl_beam)
                excl_beam_list.append(excl_beam)
                incl_beam_list.append(incl_beam)

            final_info_incl, final_info_excl = self.combine_beams(incl_beam_list, excl_beam_list)
            final_incl_beam = GrowInfoBeam(width=self.data_info.beam_width)
            final_excl_beam = GrowInfoBeam(width=self.data_info.beam_width)
            for info in final_info_incl:
                final_incl_beam.update(info, info["normalized_gain_incl"])
            for info in final_info_excl:
                final_excl_beam.update(info, info["normalized_gain_excl"])


            if len(final_incl_beam.gains) == 0 and len(final_excl_beam.gains) == 0:
                break

            stop_condition_element = self.calculate_stop_condition_element(final_incl_beam, final_excl_beam, previous_best_gain, previous_best_excl_gain)
            if stop_condition_element:
                counter_worse_best_gain = counter_worse_best_gain + 1
            else:
                counter_worse_best_gain = 0

            if len(final_incl_beam.gains) > 0:
                previous_best_gain = np.max(final_incl_beam.gains)
            if len(final_excl_beam.gains) > 0:
                previous_best_excl_gain = np.max(final_excl_beam.gains)

            if counter_worse_best_gain > k_consecutively:
                break
            else:
                rules_for_next_iter = extract_rules_from_beams([final_excl_beam, final_incl_beam])
                rules_candidates.extend(rules_for_next_iter)

        which_best_ = np.argmax([r.incl_gain_per_excl_coverage for r in rules_candidates])
        return rules_candidates[which_best_]




