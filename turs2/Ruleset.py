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
        if self.data_info.log_learning_process:
            log_folder_name = self.data_info.log_folder_name

        log_info_ruleset = ""
        for iter in range(max_iter):
            log_info_ruleset += "  Iteration: "+ str(iter) + "\n" \
                                "  ruleset total cl: " + str(self.total_cl) + "\n" \
                                "  cl data: " + str(self.cl_data) + "\n" \
                                "  cl model: "+ str(self.cl_model) + "\n" \
                                "  self.allrules_regret: " + str(self.data_encoding.allrules_regret) + "\n" \
                                "  self.allrules_cl_data: " + str(self.allrules_cl_data) + "\n\n"
            if printing:
                print("iteration ", iter)
            rule_to_add, incl_normalized_gain = self.find_next_rule(constraints=self.constraints)

            if self.data_info.rf_oob_decision_ is not None:
                rf_oob = np.array(self.data_info.rf_oob_decision_)

                if self.data_info.num_class == 2:
                    rf_oob[self.uncovered_indices] = np.median(self.data_info.rf_oob_decision_[self.uncovered_indices])
                    rf_oob[rule_to_add.bool_array] = np.median(self.data_info.rf_oob_decision_[rule_to_add.bool_array])
                    auc_flatten_else_rule = roc_auc_score(self.data_info.target, rf_oob)
                    auc_flatten_else_rule_and_rule_to_add = roc_auc_score(self.data_info.target, rf_oob)
                else:
                    rf_oob[self.uncovered_indices] = np.mean(self.data_info.rf_oob_decision_[self.uncovered_indices], axis=0)
                    rf_oob[rule_to_add.bool_array] = np.mean(self.data_info.rf_oob_decision_[rule_to_add.bool_array], axis=0)

                    auc_flatten_else_rule = roc_auc_score(self.data_info.target, rf_oob, multi_class="ovr")
                    auc_flatten_else_rule_and_rule_to_add = roc_auc_score(self.data_info.target, rf_oob, multi_class="ovr")

                # TODO: choose which one to use
                add_to_ruleset = (auc_flatten_else_rule_and_rule_to_add > auc_flatten_else_rule) or (incl_normalized_gain > 0)
                # add_to_ruleset = (incl_normalized_gain > 0)
            else:
                add_to_ruleset = (incl_normalized_gain > 0)

            if add_to_ruleset:
                self.add_rule(rule_to_add)
                if self.data_info.log_learning_process:
                    local_predict_res = get_rule_local_prediction_for_unseen_data(ruleset=self,
                                                                                  X_test=self.data_info.X_test,
                                                                                  y_test=self.data_info.y_test)
                    log_info_ruleset += "Add rule: " + get_readable_rule(rule_to_add) + \
                                        "\n Local information on test data: \n" + \
                                        "Probability: " + str(local_predict_res["rules_test_p"][iter]) + \
                                        "   Coverage: " + str(local_predict_res["rules_test_coverage"][iter]) + \
                                        "   \n Else rule probability and coverage: " + \
                                        str([local_predict_res["else_rule_p"],
                                             local_predict_res["else_rule_coverage"]]) + \
                                        "\n\n"
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

    def find_next_rule(self, rule_given=None, constraints=None):
        if rule_given is None:
            rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=self, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf, icols_in_order=[])
        else:
            rule = rule_given
        rule_to_add = rule

        excl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0)]
        excl_beam_list[0].update(rule=rule, gain=rule.excl_normalized_gain)
        incl_beam_list = []

        # TODO: store the cover of the rules as a bit string (like CLASSY's implementation) and then do diverse search.
        # now we start the real search
        previous_excl_beam = excl_beam_list[0]
        previous_incl_beam = Beam(width=self.data_info.beam_width, rule_length=0)

        for i in range(self.data_info.max_rule_length):
            current_incl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)
            current_excl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)

            for rule in previous_incl_beam.rules + previous_excl_beam.rules:
                excl_res, incl_res = rule.grow_incl_and_excl(constraints=constraints)
                if excl_res is None or incl_res is None:  # NOTE: will only return none when all data points have the same feature values in all dimensions
                    continue

                # if incl_res.incl_normalized_gain > 0:
                # TODO: whether to constrain all excl_grow_res have positive normalized gain?
                # current_incl_beam.update(incl_res,
                #                          incl_res.incl_normalized_gain)
                # TODO: this is a temporary solution for using absolute gain for incl_grow
                current_incl_beam.update(incl_res,
                                         incl_res.incl_normalized_gain / incl_res.coverage_excl)

                if rule in current_incl_beam.rules:
                    pass
                else:
                    # if excl_res.excl_normalized_gain > 0:
                    current_excl_beam.update(excl_res,
                                             excl_res.excl_normalized_gain)

            if len(current_excl_beam.rules) > 0 or len(current_incl_beam.rules) > 0:   # Can change to some other (early) stopping criteria;
                previous_excl_beam = current_excl_beam
                previous_incl_beam = current_incl_beam
                excl_beam_list.append(current_excl_beam)
                incl_beam_list.append(current_incl_beam)
            else:
                break

        log_nextbestrule = ""
        best_incl_normalized_gain = -np.inf

        # if self.data_info.log_learning_process:
        #     with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +"_incl.pickle", "wb") as pick_rule:
        #         pickle.dump(incl_beam_list, pick_rule)
        #
        #     with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +"_excl.pickle", "wb") as pick_rule:
        #         pickle.dump(excl_beam_list, pick_rule)


        for incl_beam in incl_beam_list:
            for r in incl_beam.rules:
                # incl_normalized_gain = r.incl_normalized_gain
                # TODO: a temporary solution for using absolute gain for choosing the cut point for incl_rule_grow
                incl_normalized_gain = r.incl_normalized_gain / len(r.indices_excl_overlap)

                # gain_data = self.cl_data - self.data_encoding.get_cl_data_incl(self, r, np.ones(r.coverage_excl, dtype=bool), np.ones(r.coverage, dtype=bool))
                # gain_model = self.cl_model - self.allrules_cl_model - r.cl_model - universal_code_integers(len(self.rules) + 1)
                #
                # growth_validity = check_validity_growth(r1=r.rule_base, r2=r)

                # print(get_readable_rule(r))
                # print("gain_data, gain_model: ", [gain_data, gain_model], "\n\n")
                # print("total_cl: ", self.cl_model + self.cl_data)
                # rule_condition = get_readable_rule(r)
                # self.rules_finalround_total_cl[rule_condition] = \
                #     self.data_encoding.get_cl_data_incl(self, r, np.ones(r.coverage_excl, dtype=bool), np.ones(r.coverage, dtype=bool)) + \
                #     self.allrules_cl_model + r.cl_model + universal_code_integers(len(self.rules) + 1)


                if self.data_info.log_learning_process:
                    rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r, X_test=self.data_info.X_test, y_test=self.data_info.y_test)

                    if self.data_info.rf_oob_decision_ is None:
                        oob_flatten_roc_auc = "Not available"
                        rf_original_oob_roc_auc = "Not available"
                    else:
                        oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)

                        if self.data_info.num_class == 2:
                            oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
                            oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
                            rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_))
                        else:
                            oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
                            oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
                            rf_original_oob_roc_auc = str(roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_,
                                                                        multi_class="ovr"))

                    else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
                                             self.data_info.num_class)

                    rule_prob_diff_train_test_max = np.max(abs(rule_test_p - r.prob))

                    log_nextbestrule += "\n\n************Checking rule: \n" + get_readable_rule(r) + "\n" + \
                                        "with incl_normalized_gain: " + str(r.incl_normalized_gain) + " **/** " + str(incl_normalized_gain) + \
                                        "\n with probability/coverage on test_set: " + str([rule_test_p, rule_test_coverage]) + \
                                        "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(oob_flatten_roc_auc) + \
                                        "\n with RF out-of-sample original ROC-AUC being: " + rf_original_oob_roc_auc + \
                                        "\n with else-rule training prob: " + str(else_rule_p) + \
                                        "\n with maximum rule_prob_diff_train/test: " + str(rule_prob_diff_train_test_max)

                # if incl_normalized_gain > best_incl_normalized_gain and growth_validity > 0:
                if incl_normalized_gain > best_incl_normalized_gain:
                    rule_to_add = r
                    best_incl_normalized_gain = incl_normalized_gain
                    log_nextbestrule += "\n ======== update rule_to_add ======="

        if self.data_info.log_learning_process:
            log_nextbestrule += "\n\n\n\n%%%%%%%%%%%%%%%%checking excl_beam%%%%%%%%%%%%%%%%%%\n\n\n"

            for excl_beam in excl_beam_list:
                for r in excl_beam.rules:
                    rule_test_p, rule_test_coverage = get_rule_local_prediction_for_unseen_data_this_rule_only(rule=r,
                                                                                                               X_test=self.data_info.X_test,
                                                                                                               y_test=self.data_info.y_test)
                    if self.data_info.rf_oob_decision_ is None:
                        oob_flatten_roc_auc = "Not available "
                        rf_original_oob_roc_auc = "Not available "
                    else:
                        oob_deci_flatten = np.array(self.data_info.rf_oob_decision_)
                        if self.data_info.num_class == 2:
                            oob_deci_flatten[r.bool_array] = np.median(oob_deci_flatten[r.bool_array])
                            oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten)
                            rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_)
                        else:
                            oob_deci_flatten[r.bool_array] = np.mean(oob_deci_flatten[r.bool_array], axis=0)
                            oob_flatten_roc_auc = roc_auc_score(self.data_info.target, oob_deci_flatten, multi_class="ovr")
                            rf_original_oob_roc_auc = roc_auc_score(self.data_info.target, self.data_info.rf_oob_decision_, multi_class="ovr")

                    else_rule_p = calc_probs(self.data_info.target[self.uncovered_bool & (~r.bool_array)],
                                             self.data_info.num_class)

                    log_nextbestrule += "\n\n************Checking rule: \n" + get_readable_rule(r) + "\n" + \
                                        " with coverage_excl: " + str(len(r.indices_excl_overlap)) + \
                                        "with excl_normalized_gain: " + str(r.excl_normalized_gain) + \
                                        "\n with probability/coverage on test_set: " + str(
                        [rule_test_p, rule_test_coverage]) + \
                                        "\n with RF out-of-sample ROC-AUC when 'squeezing' this rule: " + str(
                        oob_flatten_roc_auc) + \
                                        "\n with RF out-of-sample original ROC-AUC being: " + str(
                        rf_original_oob_roc_auc) + \
                                        "\n with else-rule training prob: " + str(else_rule_p)
        if self.data_info.log_learning_process:
            system_name = platform.system()
            if system_name == "Windows":
                with open(self.data_info.log_folder_name + "\\rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
                    flog_ruleset.write(log_nextbestrule)
            else:
                with open(self.data_info.log_folder_name + "/rule" + str(len(self.rules)) +".txt", "w") as flog_ruleset:
                    flog_ruleset.write(log_nextbestrule)

        return [rule_to_add, best_incl_normalized_gain]

    def evaluate_rule(self, rule):
        # TODO: check if this method still being used??
        scores_all_mgs = []
        for mg in self.modelling_groups:
            mg_and_rule_score = mg.evaluate_rule(rule)
            scores_all_mgs.append(mg_and_rule_score)

        rule_and_else_bool = np.bitwise_and(rule.bool_array, self.uncovered_bool)
        coverage_rule_and_else = np.count_nonzero(rule_and_else_bool)
        p_rule_and_else = calc_probs(self.data_info.target[rule_and_else_bool], self.data_info.num_class)
        negloglike_rule_and_else = -coverage_rule_and_else * np.sum(p_rule_and_else[rule.prob!=0] * np.log2(rule.prob[rule.prob!=0]))  # using the rule's probability
        reg_rule_and_else = rule.regret

        else_new_bool = np.bitwise_and(~rule.bool_array, self.uncovered_bool)
        coverage_else_new = np.count_nonzero(else_new_bool)
        p_else_new = calc_probs(self.data_info.target[else_new_bool], self.data_info.num_class)
        negloglike_else_new = calc_negloglike(p_else_new, coverage_else_new)
        reg_else_new = regret(coverage_else_new, self.data_info.num_class)

        total_negloglike_including_else_rule = np.sum(scores_all_mgs) + negloglike_else_new + negloglike_rule_and_else
        reg_excluding_all_rules_in_ruleset = reg_else_new + reg_rule_and_else
        rule_cl_model = rule.cl_model

        return {"total_negloglike_including_else_rule": total_negloglike_including_else_rule,
                "reg_excluding_all_rules_in_ruleset": reg_excluding_all_rules_in_ruleset,
                "rule_cl_model": rule_cl_model}

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

    def _print(self):
        readables = []
        for rule in self.rules:
            readable = ""
            which_variables = np.where(rule.condition_count != 0)[0]
            for i, v in enumerate(which_variables):
                cut = rule.condition_matrix[:, v][::-1]
                icol_name = self.data_info.feature_names[v]
                if i == len(which_variables) - 1:
                    readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + "   ===>   "
                else:
                    readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + "   &   "

            readable += "Prob Neg/Pos: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)
            print(readable)
            readables.append(readable)

        readable = "Else-rule, Prob Neg/Pos: " + str(self.else_rule_p) + ", Coverage: " + str(
            self.else_rule_coverage)
        readables.append(readable)
        print(readable)




