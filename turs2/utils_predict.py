import numpy as np
from turs2.utils_calculating_cl import *


def get_rule_local_prediction_for_unseen_data_this_rule_only(rule, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()
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
    rule_test_p = calc_probs(y_test[rule_cover], rule.ruleset.data_info.num_class)
    rule_test_coverage = np.count_nonzero(rule_cover)

    return [rule_test_p, rule_test_coverage]


def get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()
    rules_test_p = []
    rules_test_coverage = []

    allrules_cover = np.zeros(len(X_test), dtype=bool)
    rule_cover_test = []
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

        rule_cover_test.append(rule_cover)
        # only for comparison with the estimated probability from the training set, not for evaluating the model!!!
        rule_test_p = calc_probs(y_test[rule_cover], ruleset.data_info.num_class)
        rules_test_p.append(rule_test_p)
        rules_test_coverage.append(np.count_nonzero(rule_cover))

        allrules_cover = np.bitwise_or(allrules_cover, rule_cover)
    else_rule_cover = ~allrules_cover
    else_rule_coverage = np.count_nonzero(else_rule_cover)
    else_rule_p = calc_probs(y_test[else_rule_cover], ruleset.data_info.num_class)
    return {"rules_test_p": rules_test_p, "rules_test_coverage": rules_test_coverage,
            "else_rule_p": else_rule_p, "else_rule_coverage": else_rule_coverage,
            "allrules_cover": allrules_cover, "rule_cover_test": rule_cover_test}


def predict_ruleset(ruleset, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()
    prob_predicted = np.zeros((len(X_test), ruleset.data_info.num_class), dtype=float)

    rules_test_res = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rules_test_p = rules_test_res["rules_test_p"]
    allrules_cover = rules_test_res["allrules_cover"]

    allmgs_cover = np.zeros(len(X_test), dtype=bool)
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
        allmgs_cover = np.bitwise_or(allmgs_cover, mg_cover)

    not_covered_by_all_mgs = np.bitwise_and(allrules_cover, ~allmgs_cover)
    rule_cover_test = rules_test_res["rule_cover_test"]
    if any(not_covered_by_all_mgs):
        not_covered_by_all_mgs_indices = np.where(not_covered_by_all_mgs)[0]
        weighted_p = 0
        count = 0
        for i in not_covered_by_all_mgs_indices:
            rules_involve_bool = np.zeros(len(ruleset.rules), dtype=bool)
            for j, rule_j_cover in enumerate(rule_cover_test):
                if rule_j_cover[i]:
                    rules_involve_bool[j] = True
                    weighted_p += ruleset.rules[j].prob * ruleset.rules[j].coverage
                    count += ruleset.rules[j].coverage
            weighted_p = weighted_p / count
            prob_predicted[i] = weighted_p

    prob_predicted[~rules_test_res["allrules_cover"]] = ruleset.else_rule_p

    return [prob_predicted, rules_test_p, ~allrules_cover]


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
