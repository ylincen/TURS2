import numpy as np
import pickle

from turs2.Rule import *
from turs2.Beam import *
from turs2.ModellingGroup import *
from turs2.DataInfo import *
from turs2.utils_readable import *


class Ruleset:
    def __init__(self, data_info, data_encoding, model_encoding):
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

    def add_rule(self, rule):
        self.rules.append(rule)
        self.cl_data, self.allrules_cl_data = \
            self.data_encoding.update_ruleset_and_get_cl_data_ruleset_after_adding_rule(ruleset=self, rule=rule)
        self.cl_model = \
            self.model_encoding.cl_model_after_growing_rule_on_icol(rule=None, ruleset=self, icol=None, cut_option=None)

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
                                bool_cover=np.bitwise_and(rule.bool_array, self.uncovered_bool),
                                bool_use_for_model=rule.bool_array,
                                rules_involved=[len(self.rules) - 1], prob_model=rule.prob,
                                prob_cover=rule.prob_excl)
            all_mg_neglolglike.append(mg.negloglike)
            self.modelling_groups.append(mg)
        return np.sum(all_mg_neglolglike)

    def fit(self, max_iter=1000, printing=True):
        for iter in range(max_iter):
            if printing:
                print("iteration ", iter)
            rule_to_add, incl_normalized_gain = self.find_next_rule()

            if incl_normalized_gain > 0:
                if printing:
                    print(rule_to_add._print())
                    print("incl_normalized_gain:", incl_normalized_gain, "coverage_excl: ", rule_to_add.coverage_excl)
                self.add_rule(rule_to_add)
            else:
                break

    # def fit_for_tuning_cl(self, file_name, number_of_rules, max_iter=1000):
    #     for iter in range(max_iter):
    #         print("iteration ", iter)
    #         rule_to_add, incl_normalized_gain = self.find_next_rule()
    #         print(rule_to_add._print())
    #         print("incl_normalized_gain:", incl_normalized_gain, "coverage_excl: ", rule_to_add.coverage_excl)
    #         if incl_normalized_gain > 0 or len(self.rules) < number_of_rules:
    #             self.add_rule(rule_to_add)
    #             with open(file_name + str(iter) + '.pickle', 'wb') as file:
    #                 # Dump the Python object into the file
    #                 pickle.dump(self, file)
    #         else:
    #             break

    def find_next_rule(self, rule_given=None):
        if rule_given is None:
            rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
                        data_info=self.data_info, rule_base=None,
                        condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                        ruleset=self, excl_normalized_gain=-np.Inf, incl_normalized_gain=-np.Inf)
            rule_to_add = rule
        else:
            rule = rule_given

        excl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0)]
        excl_beam_list[0].update(rule=rule, gain=rule.excl_normalized_gain)
        incl_beam_list = []

        # TODO: store the cover of the rules as a bit string (like CLASSY's implementation) and then do diverse search.
        # now we start the real search
        previous_excl_beam = excl_beam_list[0]
        previous_incl_beam = Beam(width=self.data_info.beam_width, rule_length=0)

        for i in range(self.data_info.max_rule_length - 1):
            current_incl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)
            current_excl_beam = Beam(width=self.data_info.beam_width, rule_length=i + 1)

            for rule in previous_incl_beam.rules + previous_excl_beam.rules:
                excl_res, incl_res = rule.grow_incl_and_excl()
                if excl_res is None or incl_res is None:  # NOTE: will only return none when all data points have the same feature values in all dimensions
                    continue
                current_incl_beam.update(incl_res,
                                         incl_res.incl_normalized_gain)  # TODO: whether to constrain all excl_grow_res have positive normalized gain?
                if np.array_equal(incl_res.bool_array, excl_res.bool_array):
                    pass
                else:
                    current_excl_beam.update(excl_res,
                                             excl_res.excl_normalized_gain)

            if len(current_excl_beam.rules) > 0 or len(current_incl_beam.rules) > 0:   # Can change to some other (early) stopping criteria;
                previous_excl_beam = current_excl_beam
                previous_incl_beam = current_incl_beam
                excl_beam_list.append(current_excl_beam)
                incl_beam_list.append(current_incl_beam)
            else:
                break

        best_incl_normalized_gain = -np.inf
        for incl_beam in incl_beam_list:
            for r in incl_beam.rules:
                # scores = self.evaluate_rule(r)
                #
                # cl_number_of_rules = self.data_info.cl_model["l_number_of_rules"][len(self.rules) + 1]
                # cl_permutations_of_rules_candidate = math.lgamma(len(self.rules) + 2) / np.log(2)  # math.lgamma(k + 1) = np.log(math.factorial(k))
                #
                # cl_extra_cost_random_design = np.log2(self.else_rule_coverage) + self.cl_cost_random_design
                #
                # incl_absolute_gain = self.total_cl - scores["total_negloglike_including_else_rule"] - self.allrules_regret - \
                #                      scores["reg_excluding_all_rules_in_ruleset"] - scores["rule_cl_model"] - self.allrules_cl_model - \
                #                      cl_number_of_rules + cl_permutations_of_rules_candidate - cl_extra_cost_random_design
                # incl_normalized_gain = incl_absolute_gain / r.coverage_excl
                incl_normalized_gain = r.incl_normalized_gain

                if incl_normalized_gain > best_incl_normalized_gain:
                    rule_to_add = r
                    best_incl_normalized_gain = incl_normalized_gain

        return [rule_to_add, best_incl_normalized_gain]

    def evaluate_rule(self, rule):
        scores_all_mgs = []
        for mg in self.modelling_groups:
            mg_and_rule_score = mg.evaluate_rule(rule)
            scores_all_mgs.append(mg_and_rule_score)

        rule_and_else_bool = np.bitwise_and(rule.bool_array, self.uncovered_bool)
        coverage_rule_and_else = np.count_nonzero(rule_and_else_bool)
        p_rule_and_else = calc_probs(self.data_info.target[rule_and_else_bool], self.data_info.num_class)
        negloglike_rule_and_else = -coverage_rule_and_else * np.sum(p_rule_and_else[rule.prob!=0] * np.log2(rule.prob[rule.prob!=0]))  # using the rule's probability
        # reg_rule_and_else = regret(coverage_rule_and_else, self.data_info.num_class)
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
        new_ruleset = self.ruleset_after_deleting_a_rule(rule_to_modify_index)
        new_rule = self.rules[rule_to_modify_index].new_rule_after_deleting_condition(cols_to_delete_in_rule, new_ruleset)
        rule_to_add, incl_normalized_gain = new_ruleset.find_next_rule(rule_given=new_rule)
        if incl_normalized_gain > 0:
            if printing:
                print(rule_to_add._print())
                print("incl_normalized_gain:", incl_normalized_gain, "coverage_excl: ", rule_to_add.coverage_excl)
            new_ruleset.add_rule(rule_to_add)
            return new_ruleset
        else:
            return self




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

    def ruleset_after_deleting_a_rule(self, rule_to_delete):
        new_ruleset = Ruleset(self.data_info, self.data_encoding, self.model_encoding)
        for i, r in enumerate(self.rules):
            if i == rule_to_delete:
                continue
            else:
                new_ruleset.add_rule(r)
        return new_ruleset



