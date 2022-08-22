import copy

import numpy as np

from utils import *
from nml_regret import *
from newBeam import *
from newRule import *
import surrogate_tree


class ModelingGroupSet:
    # __slots__ = ["features", "target", "data_info", "modeling_group_set", "else_rule_modeling_group",
    #              "rules"]
    def __init__(self, else_rule, data_info, rules, target, features):
        self.modeling_group_set = []
        self.data_info = data_info

        self.else_rule_modeling_group = ModelingGroup(data_info, rules, target, features)
        self.else_rule_modeling_group.init_as_else_rule(else_rule)

        self.rules = rules

    def update(self, rule):
        new_modeling_group_list = []
        for i, modeling_group in enumerate(self.modeling_group_set):
            # modeling_group will be updated, and a new_modeling_group will also be returned
            new_modeling_group = modeling_group.update(rule)
            if new_modeling_group is None:
                pass
            else:
                new_modeling_group_list.append(new_modeling_group)

        new_else_modeling_group = self.else_rule_modeling_group.update_else_rule_modeling_group(rule)
        if new_else_modeling_group is None:
            pass
        else:
            new_modeling_group_list.append(new_else_modeling_group)

        self.modeling_group_set.extend(new_modeling_group_list)

    def evaluate_rule(self, rule):
        neglog_likelihood = 0
        for i, modeling_group in enumerate(self.modeling_group_set):
            neglog_likelihood = neglog_likelihood + modeling_group.evaluate_rule(rule)

        else_surrogate_score = self.else_rule_modeling_group.evaluate_rule_for_ElseRuleModelingGroup(rule,
                                                                                                     surrogate=True)
        return else_surrogate_score + neglog_likelihood + rule.regret

    def total_surrogate_score(self):
        neglog_likelihoods = [modeling_group.neglog_likelihood for modeling_group in self.modeling_group_set]
        regrets = [rule.regret for rule in self.rules]
        surrogate_score_else_rule = self.else_rule_modeling_group.surrogate_score
        return sum(neglog_likelihoods) + sum(regrets) + surrogate_score_else_rule

    def total_score(self):
        neglog_likelihoods = [modeling_group.neglog_likelihood for modeling_group in self.modeling_group_set]
        regrets = [rule.regret for rule in self.rules]
        else_rule_score = self.else_rule_modeling_group.non_surrogate_score
        return sum(neglog_likelihoods) + sum(regrets) + else_rule_score


class ModelingGroup:
    # __slots__ = ["features", "target", "data_info", "rules", "rules_involved_boolean",
    #              "rules_used_boolean", "instances_covered_boolean", "instances_modeling_boolean",
    #              "p", "neglog_likelihood", "length", "surrogate_score", "non_surrogate_score",
    #              "surrogate_equal_nonsurrogate"]
    def __init__(self, data_info, rules, target, features):
        # ***ALL*** rules in the rule set, for checking nestedness, i.e., whether one rule fully covers another rule;
        self.rules = rules

        # instances in this modeling_group are covered by these rules, represented by boolean array
        self.rules_involved_boolean = []
        # instances in this modeling_group are MODELED by these rules (i.e., nestedness removed),
        # represented by boolean array
        self.rules_used_boolean = []

        # instances IN this modeling group
        self.instances_covered_boolean = np.zeros(data_info.nrow, dtype=bool)
        # instances USED for ML estimator for this modeling group
        self.instances_modeling_boolean = np.zeros(data_info.nrow, dtype=bool)

        self.data_info = data_info
        self.p = None
        self.neglog_likelihood = None
        self.length = 0  # the length of the self.rules_invovled_boolean

        self.target = target  # target vector of the ORIGINAL DATASET, for evaluating rules
        self.features = features  # feature matrix of the ORIGINAL DATASET, for obtaining surrogate scores.

        # Below are all only for the else_rule_modeling_group
        self.surrogate_score = None
        self.non_surrogate_score = None
        self.surrogate_equal_nonsurrogate = False  # if false, we should stop adding new rules to the rule set.

    def duplicate(self):
        new_modeling_group = ModelingGroup(self.data_info, self.rules, self.target, self.features)

        new_modeling_group.rules_involved_boolean = copy.deepcopy(self.rules_involved_boolean)
        new_modeling_group.rules_used_boolean = copy.deepcopy(self.rules_used_boolean)
        new_modeling_group.instances_covered_boolean = copy.deepcopy(self.instances_covered_boolean)
        new_modeling_group.instances_modeling_boolean = copy.deepcopy(self.instances_modeling_boolean)
        new_modeling_group.p = self.p
        new_modeling_group.neglog_likelihood = self.neglog_likelihood
        new_modeling_group.length = self.length
        new_modeling_group.surrogate_score = self.surrogate_score
        new_modeling_group.non_surrogate_score = self.non_surrogate_score
        new_modeling_group.surrogate_equal_nonsurrogate = self.surrogate_equal_nonsurrogate

        return new_modeling_group

    def update(self, rule):
        """
        Update the modeling group by splitting the modeling group into two parts, with one part overlapping with
        the rule, and the other part not overlapping with the rule;
        :param rule:
        :return:
        """
        intersection_boolarray = np.bitwise_and(rule.bool_array, self.instances_covered_boolean)
        intersection_count = np.count_nonzero(intersection_boolarray)
        if intersection_count > 0:
            # modeling_group_to_add = copy.deepcopy(self)
            modeling_group_to_add = self.duplicate()
            self.extend_one(intersection_boolarray=intersection_boolarray, rule=rule, extend_with_true=True)
            modeling_group_to_add.extend_one(intersection_boolarray=intersection_boolarray, rule=rule,
                                             extend_with_true=False)
            return modeling_group_to_add
        else:
            # cover_ and model_ boolean are not changed, and hence also self.p and self.neglog_likelihood.
            self.rules_used_boolean.append(False)
            self.rules_involved_boolean.append(False)
            self.length += 1
            return None

    def extend_one(self, intersection_boolarray, rule, extend_with_true):
        """
        Extend self.rules_involved and self.rules_used, and also update self.instances_covered_boolean
        and self.instances_modeling_boolean accordingly.
        :param intersection_boolarray: equal to np.bitwise_and(rule.bool_array, self.instances_covered_boolean)
        :param rule: rule to be added
        :param extend_with_true: append True or False to self.rules_involved_boolean
        :return: None
        """
        if extend_with_true:
            self.rules_involved_boolean.append(True)
            self.rules_used_boolean.append(True)
            nest_updated = self.check_and_update_nestedness(rule)  # update self.rules_used_boolean and return a flag;

            self.instances_covered_boolean = np.bitwise_and(self.instances_covered_boolean, intersection_boolarray)
            if nest_updated is None:
                self.instances_modeling_boolean = np.bitwise_or(self.instances_modeling_boolean, rule.bool_array)
            else:
                self.instances_modeling_boolean = np.zeros(rule.data_info.nrow, dtype=bool)
                for i, r in enumerate(self.rules):
                    if self.rules_used_boolean[i]:
                        self.instances_modeling_boolean = np.bitwise_or(r.bool_array, rule.bool_array)
            self.p = calc_probs(self.target[self.instances_modeling_boolean], self.data_info.num_class)
            p_cover = calc_probs(self.target[self.instances_covered_boolean], self.data_info.num_class)

            self.neglog_likelihood = -np.sum(np.log2(self.p[self.p != 0]) * p_cover[self.p != 0]) * \
                                     np.count_nonzero(self.instances_covered_boolean)

        else:
            self.rules_involved_boolean.append(False)
            self.rules_used_boolean.append(False)

            self.instances_covered_boolean = np.bitwise_and(self.instances_covered_boolean, (~intersection_boolarray))
            # Note: self.instances_modeling_boolean unchanged, and hence also self.p

            p_cover = calc_probs(self.target[self.instances_covered_boolean], self.data_info.num_class)
            self.neglog_likelihood = -np.sum(np.log2(self.p[self.p != 0]) * p_cover[self.p != 0]) * \
                                     np.count_nonzero(self.instances_covered_boolean)

        self.length += 1

    def check_and_update_nestedness(self, rule):
        """
        Check whether the rule is fully covered by rules involved in self, and update self.rules_used_boolean
        accordingly;
        :param rule:
        :return:
        """
        nest_updated = None
        for i, r in enumerate(self.rules):
            if self.rules_used_boolean[i]:
                if np.array_equal(np.bitwise_and(rule.bool_array, r.bool_array), r.bool_array):
                    self.rules_used_boolean[-1] = False
                    nest_updated = -1
                elif np.array_equal(np.bitwise_and(rule.bool_array, r.bool_array), rule.bool_array):
                    self.rules_used_boolean[i] = False
                    nest_updated = i
                else:
                    pass
        return nest_updated

    def evaluate_nestedness(self, rule):
        """
        Evaluated whether the rule fully covers, or is fully covered by, any rule in self.rules_used_boolean
        :param rule:
        :return: -1 ("rule" is fully covered) or the index of the rule in self.rules_used_boolean is fully
        covered by "rule".
        """
        nest_to_updated = []
        for i, r in enumerate(self.rules):
            if self.rules_used_boolean[i]:
                if np.array_equal(np.bitwise_and(rule.bool_array, r.bool_array), r.bool_array):
                    nest_to_updated.append(-1)
                    break
                elif np.array_equal(np.bitwise_and(rule.bool_array, r.bool_array), rule.bool_array):
                    nest_to_updated.append(i)
                else:
                    pass
        return nest_to_updated

    def evaluate_rule(self, rule):
        """
        Check whether this rule and "self" has overlap: if yes, calculate the neglog_likelihood when splitting
        "self" into two modeling_groups, i.e., *****0 and *****1 (assuming the involved_rules for self is *****);
        if no, return the self.neglog_likelihood.
        :param rule: the rule to be evaluated.
        :return:
        """
        intersection_boolarray = np.bitwise_and(rule.bool_array, self.instances_covered_boolean)
        intersection_count = np.count_nonzero(intersection_boolarray)
        if intersection_count == 0:
            return 0
        else:
            # Those covered by both the modeling_group and the rule
            instances_part1_covered_boolean = intersection_boolarray
            nest_to_update = self.evaluate_nestedness(rule)  # needed for obtaining the modeling_boolean
            if len(nest_to_update) == 0:
                instances_part1_modeling_boolean = np.bitwise_or(self.instances_modeling_boolean, rule.bool_array)
            elif nest_to_update[0] == -1:
                instances_part1_modeling_boolean = self.instances_modeling_boolean
            else:
                instances_part1_modeling_boolean = np.zeros(rule.data_info.nrow, dtype=bool)
                for i, r in enumerate(self.rules):
                    if self.rules_used_boolean[i] and (i not in nest_to_update):
                        instances_part1_modeling_boolean = np.bitwise_or(r.bool_array, rule.bool_array)
            p1_modeling = calc_probs(self.target[instances_part1_modeling_boolean], self.data_info.num_class)
            p1_cover = calc_probs(self.target[instances_part1_covered_boolean], self.data_info.num_class)
            p1_selector = (p1_modeling != 0)
            neglog_likelihood_part1 = -np.sum(np.log2(p1_modeling[p1_selector]) *
                                              p1_cover[p1_selector]) * np.count_nonzero(instances_part1_covered_boolean)

            # Those covered by the modeling_group but NOT the rule
            instances_part2_covered_boolean = np.bitwise_and(~rule.bool_array, self.instances_covered_boolean)
            instances_part2_modeling_boolean = self.instances_modeling_boolean

            p2_modeling = calc_probs(self.target[instances_part2_modeling_boolean], self.data_info.num_class)
            p2_cover = calc_probs(self.target[instances_part2_covered_boolean], self.data_info.num_class)
            p2_selector = (p2_modeling != 0)
            neglog_likelihood_part2 = -np.sum(np.log2(p2_modeling[p2_selector]) *
                                              p2_cover[p2_selector]) * np.count_nonzero(instances_part2_covered_boolean)
            return neglog_likelihood_part2 + neglog_likelihood_part1

    def init_as_else_rule(self, else_rule):
        self.instances_modeling_boolean = np.ones(self.data_info.nrow, dtype=bool)
        self.instances_covered_boolean = np.ones(self.data_info.nrow, dtype=bool)

        self.p = else_rule.p
        self.neglog_likelihood = else_rule.neglog_likelihood

        surrogate_else_neglog_likelihood, surrogate_else_regret = \
            surrogate_tree.get_tree_cl_individual(x_train=self.features,
                                                  y_train=self.target,
                                                  num_class=self.data_info.num_class)
        self.surrogate_score = surrogate_else_neglog_likelihood + surrogate_else_regret
        self.non_surrogate_score = self.neglog_likelihood + else_rule.regret

    def update_else_rule_modeling_group(self, rule):
        """
        Check whether rule is overlap with the else-rule;
        If yes, update the else_rule_modelling_group and also create a new modeling group as "****1"
        If no, do nothing;
        :param rule:
        :return:
        """
        intersection_boolean = np.bitwise_and(self.instances_covered_boolean, rule.bool_array)
        intersection_count = np.count_nonzero(intersection_boolean)

        if intersection_count > 0:
            # new_modeling_group: the modeling group "rule & else_rule"
            # new_modeling_group = copy.deepcopy(self)
            new_modeling_group = self.duplicate()
            new_modeling_group.instances_modeling_boolean = rule.bool_array
            new_modeling_group.instances_covered_boolean = intersection_boolean

            new_modeling_group.p = rule.prob
            p_cover = calc_probs(self.target[new_modeling_group.instances_covered_boolean],
                                 self.data_info.num_class)
            new_modeling_group.neglog_likelihood = \
                -np.sum(np.log2(new_modeling_group.p[new_modeling_group.p != 0]) *
                        p_cover[new_modeling_group.p != 0]) * intersection_count

            new_modeling_group.rules_involved_boolean.append(True)
            new_modeling_group.rules_used_boolean.append(True)

            # update the else_rule_modeling_group by remove the cover of above from the else_rule
            covered_and_modelling_boolean = np.bitwise_and(self.instances_covered_boolean, ~rule.bool_array)
            self.instances_covered_boolean = covered_and_modelling_boolean
            self.instances_modeling_boolean = covered_and_modelling_boolean  # for else-rule, these two are the same

            self.rules_involved_boolean.append(False)
            self.rules_used_boolean.append(False)

            self.p = calc_probs(self.target[self.instances_modeling_boolean], self.data_info.num_class)
            self.neglog_likelihood = -np.sum(np.log2(self.p[self.p != 0]) * self.p[self.p != 0]) * \
                                     np.count_nonzero(self.instances_covered_boolean)
            surrogate_else_neglog_likelihood, surrogate_else_regret = \
                surrogate_tree.get_tree_cl_individual(x_train=self.features[covered_and_modelling_boolean],
                                                      y_train=self.target[covered_and_modelling_boolean],
                                                      num_class=self.data_info.num_class)
            self.surrogate_score = surrogate_else_neglog_likelihood + surrogate_else_regret
            self.non_surrogate_score = self.neglog_likelihood + regret(np.count_nonzero(covered_and_modelling_boolean),
                                                                       self.data_info.num_class)

            # LATER: should we change "==" to "<="??
            if surrogate_else_neglog_likelihood == self.neglog_likelihood:
                self.surrogate_equal_nonsurrogate = True
            else:
                self.surrogate_equal_nonsurrogate = False

            return new_modeling_group
        else:
            return None  # no overlap between "rule" and the else-rule, so do nothing;

    def evaluate_rule_for_ElseRuleModelingGroup(self, rule, surrogate):
        intersection_boolean = np.bitwise_and(self.instances_covered_boolean, rule.bool_array)
        intersection_count = np.count_nonzero(intersection_boolean)

        if intersection_count > 0:
            # new_modeling_group = copy.deepcopy(self)
            new_modeling_group = self.duplicate()
            new_modeling_group.instances_modeling_boolean = rule.bool_array
            new_modeling_group.instances_covered_boolean = intersection_boolean

            p_model = rule.prob
            p_cover = calc_probs(self.target[new_modeling_group.instances_covered_boolean],
                                 self.data_info.num_class)
            neglog_likelihood = \
                -np.sum(np.log2(p_model[p_model != 0]) *
                        p_cover[p_model != 0]) * intersection_count

            covered_and_modelling_boolean = np.bitwise_and(self.instances_covered_boolean, ~rule.bool_array)

            if surrogate:
                if any(covered_and_modelling_boolean):
                    surrogate_else_neglog_likelihood, surrogate_else_regret = \
                        surrogate_tree.get_tree_cl_individual(x_train=self.features[covered_and_modelling_boolean],
                                                              y_train=self.target[covered_and_modelling_boolean],
                                                              num_class=self.data_info.num_class)
                else:
                    surrogate_else_neglog_likelihood, surrogate_else_regret = 0, 0
                return surrogate_else_neglog_likelihood + surrogate_else_regret + neglog_likelihood
            else:
                p_else = calc_probs(self.target[covered_and_modelling_boolean], self.data_info.num_class)
                else_neglog_likelihood = -np.sum(np.log2(p_else[p_else != 0]) * p_else[p_else != 0]) * \
                                         np.count_nonzero(covered_and_modelling_boolean)
                return else_neglog_likelihood + neglog_likelihood
        else:
            return None
