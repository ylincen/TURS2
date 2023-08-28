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
                                # bool_cover=np.bitwise_and(rule.bool_array, self.uncovered_bool),
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
    # def find_next_rule(self, rule_given=None, constraints=None):
    #     if rule_given is None:
    #         rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
    #                     data_info=self.data_info, rule_base=None,
    #                     condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
    #                     ruleset=self, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf, icols_in_order=[])
    #     else:
    #         rule = rule_given
    #     rule_to_add = rule
    #
    #     excl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0)]
    #     excl_beam_list[0].update(rule=rule, gain=rule.excl_normalized_gain)
    #     incl_beam_list = []
    #
    #     # TODO: store the cover of the rules as a bit string (like CLASSY's implementation) and then do diverse search.
    #     # now we start the real search
    #     previous_excl_beam = excl_beam_list[0]
    #     previous_incl_beam = Beam(width=self.data_info.beam_width, rule_length=0)
    #
    #     for i in range(self.data_info.max_rule_length):
    #         current_incl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)
    #         current_excl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)
    #
    #         for rule in previous_incl_beam.rules + previous_excl_beam.rules:
    #             excl_res, incl_res = rule.grow_incl_and_excl(constraints=constraints)
    #             if excl_res is None or incl_res is None:  # NOTE: will only return none when all data points have the same feature values in all dimensions
    #                 continue
    #
    #             # if incl_res.incl_normalized_gain > 0:
    #             # TODO: whether to constrain all excl_grow_res have positive normalized gain?
    #             # current_incl_beam.update(incl_res,
    #             #                          incl_res.incl_normalized_gain)
    #             # TODO: this is a temporary solution for using absolute gain for incl_grow
    #             current_incl_beam.update(incl_res,
    #                                      incl_res.incl_normalized_gain / incl_res.coverage_excl)
    #
    #             # if rule in current_incl_beam.rules:  # TODO: doesn't understand why I wrote this
    #             if excl_res in current_incl_beam.rules:  # TODO: check whether this is correct;
    #                 pass
    #             else:
    #                 # if excl_res.excl_normalized_gain > 0:
    #                 # current_excl_beam.update(excl_res,
    #                 #                          excl_res.excl_normalized_gain)
    #                 # TODO: this is a temporary solution for using absolute gain for excl_grow
    #                 current_excl_beam.update(excl_res,
    #                                          excl_res.excl_normalized_gain / excl_res.coverage_excl)
    #
    #         if len(current_excl_beam.rules) > 0 or len(current_incl_beam.rules) > 0:   # Can change to some other (early) stopping criteria;
    #             previous_excl_beam = current_excl_beam
    #             previous_incl_beam = current_incl_beam
    #             excl_beam_list.append(current_excl_beam)
    #             incl_beam_list.append(current_incl_beam)
    #         else:
    #             break
    #
    #     log_nextbestrule = ""
    #     best_incl_normalized_gain = -np.inf
    #
    #     pickle_save_ruleset = False
    #     if pickle_save_ruleset:
    #         with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +"_incl.pickle", "wb") as pick_rule:
    #             pickle.dump(incl_beam_list, pick_rule)
    #
    #         with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +"_excl.pickle", "wb") as pick_rule:
    #             pickle.dump(excl_beam_list, pick_rule)
    #
    #     for incl_beam in incl_beam_list:
    #         for r in incl_beam.rules:
    #             # incl_normalized_gain = r.incl_normalized_gain
    #             # TODO: a temporary solution for using absolute gain for choosing the cut point for incl_rule_grow
    #             incl_normalized_gain = r.incl_normalized_gain / len(r.indices_excl_overlap)
    #
    #             if self.data_info.log_learning_process:
    #                 rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r, X_test=self.data_info.X_test, y_test=self.data_info.y_test)
    #
    #                 if self.data_info.rf_oob_decision_ is None:
    #                     oob_flatten_roc_auc = "Not available"
    #                     rf_original_oob_roc_auc = "Not available"
    #                 else:
    #                     oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)
    #
    #                     if self.data_info.num_class == 2:
    #                         oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
    #                         rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_))
    #                     else:
    #                         oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
    #                         rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_,
    #                                                                     multi_class="ovr"))
    #
    #                 else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
    #                                          self.data_info.num_class)
    #
    #                 rule_prob_diff_train_test_max = np.max(abs(rule_test_p - r.prob))
    #
    #                 log_nextbestrule += "\n\n************Checking rule: \n" + print_(r) + "\n" + \
    #                                     "with incl_normalized_gain: " + str(r.incl_normalized_gain) + " **/** " + str(incl_normalized_gain) + \
    #                                     "\n with probability/coverage on test_set: " + str([rule_test_p, rule_test_coverage]) + \
    #                                     "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(oob_flatten_roc_auc) + \
    #                                     "\n with RF out-of-sample original ROC-AUC being: " + rf_original_oob_roc_auc + \
    #                                     "\n with else-rule training prob: " + str(else_rule_p) + \
    #                                     "\n with maximum rule_prob_diff_train/test: " + str(rule_prob_diff_train_test_max)
    #
    #             if incl_normalized_gain > best_incl_normalized_gain:
    #                 rule_to_add = r
    #                 best_incl_normalized_gain = incl_normalized_gain
    #                 log_nextbestrule += "\n ======== update rule_to_add ======="
    #
    #     if self.data_info.log_learning_process:
    #         log_nextbestrule += "\n\n\n\n%%%%%%%%%%%%%%%%checking excl_beam%%%%%%%%%%%%%%%%%%\n\n\n"
    #
    #         for excl_beam in excl_beam_list:
    #             for r in excl_beam.rules:
    #                 rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r,
    #                                                                                                            X_test=self.data_info.X_test,
    #                                                                                                            y_test=self.data_info.y_test)
    #                 if self.data_info.rf_oob_decision_ is None:
    #                     oob_flatten_roc_auc = "Not available "
    #                     rf_original_oob_roc_auc = "Not available "
    #                 else:
    #                     oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)
    #                     if self.data_info.num_class == 2:
    #                         oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
    #                         rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_)
    #                     else:
    #                         oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
    #                         rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_, multi_class="ovr")
    #
    #                 else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
    #                                          self.data_info.num_class)
    #
    #                 log_nextbestrule += "\n\n************Checking rule: \n" + print_(r) + "\n" + \
    #                                     " with coverage_excl: " + str(len(r.indices_excl_overlap)) + \
    #                                     "with excl_normalized_gain: " + str(r.excl_normalized_gain) + \
    #                                     "\n with probability/coverage on test_set: " + str(
    #                     [rule_test_p, rule_test_coverage]) + \
    #                                     "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(
    #                     oob_flatten_roc_auc) + \
    #                                     "\n with RF out-of-sample original ROC-AUC being: " + str(
    #                     rf_original_oob_roc_auc) + \
    #                                     "\n with else-rule training prob: " + str(else_rule_p)
    #     if self.data_info.log_learning_process:
    #         system_name = platform.system()
    #         if system_name == "Windows":
    #             with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
    #                 flog_ruleset.write(log_nextbestrule)
    #         else:
    #             with open(self.data_info.log_folder_name + "/rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
    #                 flog_ruleset.write(log_nextbestrule)
    #
    #     return [rule_to_add, best_incl_normalized_gain]

    # def evaluate_rule(self, rule):
    #     # TODO: check if this method still being used??
    #     scores_all_mgs = []
    #     for mg in self.modelling_groups:
    #         mg_and_rule_score = mg.evaluate_rule(rule)
    #         scores_all_mgs.append(mg_and_rule_score)
    #
    #     rule_and_else_bool = np.bitwise_and(rule.bool_array, self.uncovered_bool)
    #     coverage_rule_and_else = np.count_nonzero(rule_and_else_bool)
    #     p_rule_and_else = calc_probs(self.data_info.target[rule_and_else_bool], self.data_info.num_class)
    #     negloglike_rule_and_else = -coverage_rule_and_else * np.sum(p_rule_and_else[rule.prob!=0] * np.log2(rule.prob[rule.prob!=0]))  # using the rule's probability
    #     reg_rule_and_else = rule.regret
    #
    #     else_new_bool = np.bitwise_and(~rule.bool_array, self.uncovered_bool)
    #     coverage_else_new = np.count_nonzero(else_new_bool)
    #     p_else_new = calc_probs(self.data_info.target[else_new_bool], self.data_info.num_class)
    #     negloglike_else_new = calc_negloglike(p_else_new, coverage_else_new)
    #     reg_else_new = regret(coverage_else_new, self.data_info.num_class)
    #
    #     total_negloglike_including_else_rule = np.sum(scores_all_mgs) + negloglike_else_new + negloglike_rule_and_else
    #     reg_excluding_all_rules_in_ruleset = reg_else_new + reg_rule_and_else
    #     rule_cl_model = rule.cl_model
    #
    #     return {"total_negloglike_including_else_rule": total_negloglike_including_else_rule,
    #             "reg_excluding_all_rules_in_ruleset": reg_excluding_all_rules_in_ruleset,
    #             "rule_cl_model": rule_cl_model}

    @staticmethod
    def calculate_stop_condition_element(incl_beam, excl_beam, previous_best_gain, previous_best_excl_gain):
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
        # print("number of iterations: ", i)
        return rules_candidates[which_best_]

    # def find_next_rule_beamsearch(self, rule_given=None, constraints=None):
    #     if rule_given is None:
    #         rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
    #                     data_info=self.data_info, rule_base=None,
    #                     condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
    #                     ruleset=self, excl_mdl_gain=-np.Inf, incl_mdl_gain=-np.Inf, icols_in_order=[])
    #     else:
    #         rule = rule_given
    #     rule_to_add = rule
    #
    #     excl_beam_list = [Beam(width=self.data_info.beam_width)]
    #     excl_beam_list[0].update(rule=rule, gain=rule.excl_mdl_gain)
    #     incl_beam_list = []
    #
    #     # TODO: store the cover of the rules as a bit string (like CLASSY's implementation) and then do diverse search.
    #     # now we start the real search
    #     previous_excl_beam = excl_beam_list[0]
    #     previous_incl_beam = Beam(width=self.data_info.beam_width)
    #
    #     for i in range(self.data_info.max_grow_iter):
    #         current_incl_beam = Beam(width=self.data_info.beam_width)
    #         current_excl_beam = Beam(width=self.data_info.beam_width)
    #
    #         for rule in previous_incl_beam.rules + previous_excl_beam.rules:
    #             excl_res_beam, incl_res_beam = rule.grow_incl_and_excl_return_beam(constraints=constraints)
    #             if len(excl_res_beam) == 0 and len(incl_res_beam) == 0:
    #                 continue
    #
    #             # TODO: this is a temporary solution for using absolute gain for incl_grow
    #             # TODO: Remember to avoid the same rule to be added multiple times to the beam;
    #             for incl_res in incl_res_beam:
    #                 current_incl_beam.update(incl_res, incl_res.incl_mdl_gain / incl_res.coverage_excl)
    #
    #             for excl_res in excl_res_beam:
    #                 if excl_res in current_incl_beam.rules or excl_res in current_excl_beam.rules:
    #                     pass
    #                 else:
    #                     # TODO: this is a temporary solution for using absolute gain for excl_grow
    #                     current_excl_beam.update(excl_res,
    #                                              excl_res.excl_mdl_gain / excl_res.coverage_excl)
    #
    #         if len(current_excl_beam.rules) > 0 or len(current_incl_beam.rules) > 0:  # No (early) stopping criteria
    #             previous_excl_beam = current_excl_beam
    #             previous_incl_beam = current_incl_beam
    #             excl_beam_list.append(current_excl_beam)
    #             incl_beam_list.append(current_incl_beam)
    #         else:
    #             break
    #
    #     log_nextbestrule = ""
    #     best_incl_normalized_gain = -np.inf
    #
    #     for incl_beam in incl_beam_list:
    #         for r in incl_beam.rules:
    #             incl_normalized_gain = r.incl_mdl_gain / len(r.indices_excl_overlap)
    #             # incl_mdl_gain = self.data_encoding.get_cl_data_incl(ruleset=self, rule=r,
    #             #                                                     excl_bi_array=np.ones(r.coverage_excl, dtype=bool),
    #             #                                                     incl_bi_array=np.ones(r.coverage, dtype=bool))
    #             # incl_normalized_gain = incl_mdl_gain / len(r.indices_excl_overlap)
    #
    #             if self.data_info.log_learning_process:
    #                 rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r, X_test=self.data_info.X_test, y_test=self.data_info.y_test)
    #
    #                 if self.data_info.rf_oob_decision_ is None:
    #                     oob_flatten_roc_auc = "Not available"
    #                     rf_original_oob_roc_auc = "Not available"
    #                 else:
    #                     oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)
    #
    #                     if self.data_info.num_class == 2:
    #                         oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
    #                         rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_))
    #                     else:
    #                         oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
    #                         rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_,
    #                                                                     multi_class="ovr"))
    #
    #                 else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
    #                                          self.data_info.num_class)
    #
    #                 rule_prob_diff_train_test_max = np.max(abs(rule_test_p - r.prob))
    #
    #                 log_nextbestrule += "\n\n************Checking rule: \n" + print_(r) + "\n" + \
    #                                     "with incl_gain: " + str(r.incl_mdl_gain) + \
    #                                     "\n with probability/coverage on test_set: " + str([rule_test_p, rule_test_coverage]) + \
    #                                     "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(oob_flatten_roc_auc) + \
    #                                     "\n with RF out-of-sample original ROC-AUC being: " + rf_original_oob_roc_auc + \
    #                                     "\n with else-rule training prob: " + str(else_rule_p) + \
    #                                     "\n with maximum rule_prob_diff_train/test: " + str(rule_prob_diff_train_test_max)
    #
    #             if incl_normalized_gain > best_incl_normalized_gain:
    #                 rule_to_add = r
    #                 best_incl_normalized_gain = incl_normalized_gain
    #                 log_nextbestrule += "\n ======== update rule_to_add ======="
    #
    #     if self.data_info.log_learning_process:
    #         log_nextbestrule += "\n\n\n\n%%%%%%%%%%%%%%%%checking excl_beam%%%%%%%%%%%%%%%%%%\n\n\n"
    #
    #         for excl_beam in excl_beam_list:
    #             for r in excl_beam.rules:
    #                 rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r,
    #                                                                                                            X_test=self.data_info.X_test,
    #                                                                                                            y_test=self.data_info.y_test)
    #                 if self.data_info.rf_oob_decision_ is None:
    #                     oob_flatten_roc_auc = "Not available "
    #                     rf_original_oob_roc_auc = "Not available "
    #                 else:
    #                     oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)
    #                     if self.data_info.num_class == 2:
    #                         oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
    #                         rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_)
    #                     else:
    #                         oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
    #                         oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
    #                         rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_, multi_class="ovr")
    #
    #                 else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
    #                                          self.data_info.num_class)
    #
    #                 log_nextbestrule += "\n\n************Checking rule: \n" + print_(r) + "\n" + \
    #                                     " with coverage_excl: " + str(len(r.indices_excl_overlap)) + \
    #                                     " with excl_gain: " + str(r.excl_mdl_gain) + \
    #                                     " with cl_model: " + str(r.cl_model) + \
    #                                     "\n with probability/coverage on test_set: " + str(
    #                     [rule_test_p, rule_test_coverage]) + \
    #                                     "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(
    #                     oob_flatten_roc_auc) + \
    #                                     "\n with RF out-of-sample original ROC-AUC being: " + str(
    #                     rf_original_oob_roc_auc) + \
    #                                     "\n with else-rule training prob: " + str(else_rule_p)
    #     if self.data_info.log_learning_process:
    #         system_name = platform.system()
    #         if system_name == "Windows":
    #             with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
    #                 flog_ruleset.write(log_nextbestrule)
    #         else:
    #             with open(self.data_info.log_folder_name + "/rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
    #                 flog_ruleset.write(log_nextbestrule)
    #
    #     return [rule_to_add, best_incl_normalized_gain]


    ################################################################
    ##############Below is for Human-guided Learning################
    ################################################################
    def modify_rule(self, rule_to_modify_index, cols_to_delete_in_rule, printing=True):
        pass

    def delete_rule_i_and_search_again(self, rule_to_delete_index, printing=True):
        new_ruleset = self.ruleset_after_deleting_a_rule(rule_to_delete_index)
        new_ruleset.fit(printing=printing)
        return new_ruleset

    def modify_rule_i_other_conditions_kept(self, rule_to_modify_index, cols_to_delete_in_rule, printing=True):
        new_ruleset = self.ruleset_after_deleting_a_rule(rule_to_modify_index)
        new_rule = self.rules[rule_to_modify_index].new_rule_after_deleting_condition(cols_to_delete_in_rule, new_ruleset)

        constraints = {}
        constraints["icols_to_skip"] = cols_to_delete_in_rule
        rule_to_add, incl_normalized_gain = new_ruleset.find_next_rule(rule_given=new_rule, constraints=constraints)
        if incl_normalized_gain > 0:
            if printing:
                print(rule_to_add._print())
                print("incl_normalized_gain:", incl_normalized_gain, "coverage_excl: ", rule_to_add.coverage_excl)
            new_ruleset.add_rule(rule_to_add)
            return new_ruleset
        else:
            return self

    def modify_rule_i_other_variables_kept(self, rule_to_modify_index, icols_to_delete):
        new_ruleset = self.ruleset_after_deleting_a_rule(rule_to_modify_index)

        new_rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=new_ruleset.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=new_ruleset, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf,
                        icols_in_order=[])

        icols_to_search = [icol for icol in self.rules[rule_to_modify_index].icols_in_order if icol not in icols_to_delete]
        if len(icols_to_search) == 0:
            pass
            #

    def ruleset_after_deleting_a_rule(self, rule_to_delete):
        new_ruleset = Ruleset(self.data_info, self.data_encoding, self.model_encoding)
        for i, r in enumerate(self.rules):
            if i == rule_to_delete:
                continue
            else:
                new_ruleset.add_rule(r)
        return new_ruleset

    #################################################################################################################
    ################################# Below are all for rule list searching #########################################
    #################################################################################################################

    def add_rule_in_rulelist(self, rule):
        self.rules.append(rule)

        self.uncovered_bool = np.bitwise_and(self.uncovered_bool, ~rule.bool_array_excl)
        self.uncovered_indices = np.where(self.uncovered_bool)[0]

        self.else_rule_coverage = len(self.uncovered_indices)
        self.else_rule_p = calc_probs(self.data_info.target[self.uncovered_indices], self.data_info.num_class)
        self.else_rule_negloglike = calc_negloglike(self.else_rule_p, self.else_rule_coverage)
        self.else_rule_regret = regret(self.else_rule_coverage, self.data_info.num_class)
        self.elserule_total_cl = self.else_rule_regret + self.else_rule_negloglike

        self.cl_model += rule.cl_model
        self.allrules_regret += rule.regret
        self.allrules_neglolglike += rule.neglog_likelihood_excl

        self.negloglike = self.allrules_neglolglike + self.else_rule_negloglike
        self.regret = self.allrules_regret + self.else_rule_regret
        self.total_cl = self.negloglike + self.regret + self.cl_model

        if len(self.rules) > self.data_info.cached_number_of_rules_for_cl_model - 5: # I am not sure whether to put 1 or 2 here, so for the safe side, just do 5;
            self.data_info.cached_number_of_rules_for_cl_model = 2 * self.data_info.cached_number_of_rules_for_cl_model
            self.data_info.cl_model["l_number_of_rules"] = \
                [universal_code_integers(i) for i in range(self.data_info.cached_number_of_rules_for_cl_model)]

    def fit_rulelist(self, max_iter=1000):
        for iter in range(max_iter):
            # print("iteration ", iter)
            rule_to_add = self.find_next_rule_inrulelist()
            if rule_to_add.excl_normalized_gain > 0:
                self.add_rule_in_rulelist(rule_to_add)
            else:
                break

    def find_next_rule_inrulelist(self):
        # An empty rule
        rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
                    data_info=self.data_info, rule_base=None,
                    condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                    ruleset=self, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf)
        excl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0)]
        excl_beam_list[0].update(rule=rule, gain=rule.excl_normalized_gain)
        previous_excl_beam = excl_beam_list[0]

        rule_to_add = rule

        for i in range(self.data_info.max_rule_length - 1):
            current_excl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)

            for rule in previous_excl_beam.rules:
                excl_grow_res = rule.grow_rulelist()
                current_excl_beam.update(excl_grow_res,
                                         excl_grow_res.excl_normalized_gain)  # TODO: whether to constrain all excl_grow_res have positive normalized gain?
                if excl_grow_res.excl_normalized_gain > rule_to_add.excl_normalized_gain:
                    rule_to_add = excl_grow_res

            if len(current_excl_beam.rules) > 0:
                previous_excl_beam = current_excl_beam
                excl_beam_list.append(current_excl_beam)
            else:
                break

        return rule_to_add

    # def _print(self):
    #     readables = []
    #     for rule in self.rules:
    #         readable = ""
    #         which_variables = np.where(rule.condition_count != 0)[0]
    #         for i, v in enumerate(which_variables):
    #             cut = rule.condition_matrix[:, v][::-1]
    #             icol_name = self.data_info.feature_names[v]
    #             if i == len(which_variables) - 1:
    #                 readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + "   ===>   "
    #             else:
    #                 readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + "   &   "
    #
    #         readable += "Prob Neg/Pos: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)
    #         print(readable)
    #         readables.append(readable)
    #
    #     readable = "Else-rule, Prob Neg/Pos: " + str(self.else_rule_p) + ", Coverage: " + str(
    #         self.else_rule_coverage)
    #     readables.append(readable)
    #     print(readable)




