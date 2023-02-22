import numpy as np
from turs2.utils_calculating_cl import *


def predict_ruleset(ruleset, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()
    prob_predicted = np.zeros((len(X_test), ruleset.data_info.num_class), dtype=float)
    rules_test_p = []
    covered_all_rules = np.zeros(len(X_test), dtype=bool)  # for calculating the else_rule's cover

    for rule in ruleset.rules:
        condition_matrix = np.array(rule.condition_matrix)
        condition_count = np.array(rule.condition_count)
        which_vars = np.where(condition_count > 0)[0]

        upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
        upper_bound[np.isnan(upper_bound)] = np.Inf
        lower_bound[np.isnan(lower_bound)] = -np.Inf

        rule_cover = np.ones(len(X_test), dtype=bool)
        for v in which_vars:
            rule_cover = rule_cover & (X_test[:, v] < upper_bound[v]) & (X_test[:, v] >= lower_bound[v])

        # only for comparison with the estimated probability from the training set, not for evaluating the model!!!
        rule_test_p = calc_probs(y_test[rule_cover], ruleset.data_info.num_class)
        rules_test_p.append(rule_test_p)

    for mg in ruleset.modelling_groups:
        mg_cover = np.ones(len(X_test), dtype=bool)
        for rule_index in mg.rules_involvde:
            rule = ruleset.rules[rule_index]
            condition_matrix = np.array(rule.condition_matrix)
            condition_count = np.array(rule.condition_count)
            which_vars = np.where(condition_count > 0)[0]

            upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
            upper_bound[np.isnan(upper_bound)] = np.Inf
            lower_bound[np.isnan(lower_bound)] = -np.Inf

            for v in which_vars:
                mg_cover = mg_cover & (X_test[:, v] < upper_bound[v]) & (X_test[:, v] >= lower_bound[v])

        prob_predicted[mg_cover] = mg.prob_model
        covered_all_rules = np.bitwise_or(covered_all_rules, mg_cover)

    prob_predicted[~covered_all_rules] = ruleset.else_rule_p
    if any(~covered_all_rules):
        rules_test_p.append(calc_probs(y_test[~covered_all_rules], ruleset.data_info.num_class))
    else:
        rules_test_p.append([0, 0])
    return [prob_predicted, rules_test_p]


def predict_rulelist(ruleset, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
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