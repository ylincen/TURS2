import numpy as np

from turs2.Rule import *
from turs2.Beam import *
from turs2.ModellingGroup import *


def get_readable_rule(rule, feature_names):
    readable = ""
    which_variables = np.where(rule.condition_count != 0)[0]
    for v in which_variables:
        cut = rule.condition_matrix[:, v][::-1]
        icol_name = feature_names[v]
        readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + ";   "

    readable += "Prob: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)
    return(readable)


def get_readable_rules(ruleset):
    readables = []
    for rule in ruleset.rules:
        readable = ""
        which_variables = np.where(rule.condition_count != 0)[0]
        for v in which_variables:
            cut = rule.condition_matrix[:, v][::-1]
            icol_name = ruleset.data_info.feature_names[v]
            readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + ";   "

        readable += "Prob: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)
        readables.append(readable)

    readable = "Else-rule, Prob: " + str(ruleset.else_rule_p) + ", Coverage: " + str(ruleset.else_rule_coverage)
    readables.append(readable)
    return readables


def predict_rulelist(ruleset, X_test, y_test):
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy().flatten()
    covered = np.zeros(len(X_test), dtype=bool)
    prob_predicted = np.zeros((len(X_test), ruleset.data_info.num_class), dtype=float)

    rules_test_p = []

    for rule in ruleset.rules:
        rule_cover = ~covered

        condition_matrix = np.array(rule.condition_matrix)
        condition_count = np.array(rule.condition_count)
        which_vars = np.where(condition_count > 0)[0]

        upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
        upper_bound[np.isnan(upper_bound)] = np.Inf
        lower_bound[np.isnan(lower_bound)] = -np.Inf

        for v in which_vars:
            rule_cover = rule_cover & (X_test[:, v] < upper_bound[v]) & (X_test[:, v] >= lower_bound[v])

        rule_test_p = calc_probs(y_test[rule_cover], ruleset.data_info.num_class)
        rules_test_p.append(rule_test_p)

        prob_predicted[rule_cover] = rule.prob_excl
        covered = np.bitwise_or(covered, rule_cover)

    prob_predicted[~covered] = ruleset.else_rule_p
    if any(~covered):
        rules_test_p.append(calc_probs(y_test[~covered], ruleset.data_info.num_class))
    else:
        rules_test_p.append([0, 0])
    return [prob_predicted, rules_test_p]


class Ruleset:
    def __init__(self, data_info):
        self.rules = []

        self.default_p = calc_probs(target=data_info.target, num_class=data_info.num_class)

        self.negloglike = -np.sum(data_info.nrow * np.log2(self.default_p[self.default_p != 0]) * self.default_p[self.default_p != 0])
        self.regret = regret(data_info.nrow, data_info.num_class)
        self.cl_model = 0
        self.total_cl = self.cl_model + self.negloglike + self.regret

        self.data_info = data_info
        self.modelling_groups = [ModellingGroup(data_info=data_info)]

        self.allrules_neglolglike = 0
        self.allrules_regret = 0

        self.cl_model = 0

        self.else_rule_p = self.default_p
        self.else_rule_negloglike = self.negloglike
        self.else_rule_regret = self.regret
        self.else_rule_coverage = self.data_info.nrow
        self.elserule_total_cl = self.else_rule_regret + self.else_rule_negloglike

        self.uncovered_indices = np.arange(data_info.nrow)
        self.uncovered_bool = np.ones(self.data_info.nrow, dtype=bool)

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

    def fit_rulelist(self, max_iter=1000):
        for iter in range(max_iter):
            print("iteration ", iter)
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

        for i in range(self.data_info.max_rule_length):
            current_excl_beam = Beam(width=self.data_info.beam_width, rule_length=i+1)

            for rule in previous_excl_beam.rules:
                excl_grow_res = rule.grow_rulelist()
                current_excl_beam.update(excl_grow_res, excl_grow_res.excl_normalized_gain)  # TODO: whether to constrain all excl_grow_res have positive normalized gain?
                if excl_grow_res.excl_normalized_gain > rule_to_add.excl_normalized_gain:
                    rule_to_add = excl_grow_res

            if len(current_excl_beam.rules) > 0:
                previous_excl_beam = current_excl_beam
                excl_beam_list.append(current_excl_beam)
            else:
                break

        return rule_to_add

