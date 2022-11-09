import copy

import numpy as np

import nml_regret
from utils import *
from nml_regret import *
from newBeam import *
from newRule import *
import surrogate_tree
from ModellingGroup import *


class ModelingGroupSet:
    def __init__(self, else_rule, data_info, rules, target, features):
        """
        Initialize the modelling group set to have only one modelling group: the else_rule_modeling_group;
        Modeling_group is either an individual rule, or the overlap of multiple rules
        """
        self.modeling_group_set = []
        self.data_info = data_info

        self.else_rule_modeling_group = ModelingGroup(data_info, rules, target, features)
        self.else_rule_modeling_group.init_as_else_rule(else_rule)

        self.rules = rules

    def update(self, rule):
        """
        This function is used to update the modeling_group_set by adding one rule to it.
        That is, when we have one new rule to be added to the ruleset, we have to update our modelling group set
        accordingly.
        Specifically, we check for each modeling group, and check whether the new rule has overlap with it:
            - if no, we update the information for this modeling group (e.g., by adding the information that this new
                rule is not involved in this overlap;
            - if yes,
                - we first go over all rules involved in this modelling group and checks if any rule is fully covered,
                or fully covers this new rule. If yes, we remove the "bigger rule". If no, we take no action.
                - next, we split the modeling group into two modelling groups, one with the new rule involved,
                the other with the new rule NOT involved.
        """
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

    def evaluate_rule(self, rule, surrogate):
        # get the neglog_likelihood for each modelling_group
        neglog_likelihood = 0
        for i, modeling_group in enumerate(self.modeling_group_set):
            neglog_likelihood = neglog_likelihood + modeling_group.evaluate_rule(rule)

        # get the neglog_likelihood and the regret for the else_rule \intersection rule
        else_surrogate_score = self.else_rule_modeling_group.evaluate_rule_for_ElseRuleModelingGroup(rule,
                                                                                                     surrogate=surrogate)

        if surrogate:
            return else_surrogate_score + neglog_likelihood + rule.regret
        else:
            reg = 0
            for r in self.rules:
                reg += r.regret

            return else_surrogate_score + neglog_likelihood + rule.regret + reg

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


