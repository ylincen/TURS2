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
    rules_test_p_NotThisRule = []
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
        testdata_not_this_rule_p = calc_probs(y_test[~rule_cover], ruleset.data_info.num_class)

        rules_test_p.append(rule_test_p)
        rules_test_p_NotThisRule.append(testdata_not_this_rule_p)
        rules_test_coverage.append(np.count_nonzero(rule_cover))

        allrules_cover = np.bitwise_or(allrules_cover, rule_cover)
    else_rule_cover = ~allrules_cover
    else_rule_coverage = np.count_nonzero(else_rule_cover)
    else_rule_p = calc_probs(y_test[else_rule_cover], ruleset.data_info.num_class)

    rules_test_p_NotThisRule_including_elserule = rules_test_p_NotThisRule + \
                                                  [calc_probs(y_test[~else_rule_cover], ruleset.data_info.num_class)]
    rules_cover_test_including_elserule = rule_cover_test + [else_rule_cover]

    return {"rules_test_p": rules_test_p, "rules_test_coverage": rules_test_coverage,
            "else_rule_p": else_rule_p, "else_rule_coverage": else_rule_coverage,
            "allrules_cover": allrules_cover, "rule_cover_test": rule_cover_test,
            "rules_test_p_NotThisRule_including_elserule": rules_test_p_NotThisRule_including_elserule,
            "rules_cover_test_including_elserule": rules_cover_test_including_elserule}


def predict_ruleset(ruleset, X_test, y_test):
    if type(X_test) != np.ndarray:
        X_test = X_test.to_numpy()

    if type(y_test) != np.ndarray:
        y_test = y_test.to_numpy().flatten()

    prob_predicted = np.zeros((len(X_test), ruleset.data_info.num_class), dtype=float)
    cover_matrix = np.zeros((len(X_test), len(ruleset.rules) + 1), dtype=bool)

    test_uncovered_bool = np.ones(len(X_test), dtype=bool)
    for ir, rule in enumerate(ruleset.rules):
        r_bool_array = np.ones(len(X_test), dtype=bool)

        condition_matrix = np.array(rule.condition_matrix)
        condition_count = np.array(rule.condition_count)
        which_vars = np.where(condition_count > 0)[0]

        upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
        upper_bound[np.isnan(upper_bound)] = np.Inf
        lower_bound[np.isnan(lower_bound)] = -np.Inf

        for v in which_vars:
            r_bool_array = r_bool_array & (X_test[:, v] < upper_bound[v]) & (X_test[:, v] >= lower_bound[v])

        cover_matrix[:, ir] = r_bool_array
        test_uncovered_bool = test_uncovered_bool & ~r_bool_array
    cover_matrix[:, -1] = test_uncovered_bool

    # From chatGPT: "I use dtype=object to allow the elements of powers_of_2 and binary_vector to be Python integers which can handle arbitrary large values."
    # That is, by using "object", the element of numpy array becomes Python Int, instead of numpy.int64;
    cover_matrix_int = cover_matrix.astype(int)
    unique_id = np.zeros(len(X_test), dtype=object)
    power_of_two = 2 ** np.arange(cover_matrix_int.shape[1], dtype=object)
    for kol in range(cover_matrix_int.shape[1]):
        # unique_id += 2 ** kol * cover_matrix_int[:, kol]    # This may fail when 2 ** kol becomes very large
        unique_id += power_of_two[kol] * cover_matrix_int[:, kol].astype(object)

    groups, ret_index = np.unique(unique_id, return_index=True)
    unique_id_dir = {}
    for g, rind in zip(groups, ret_index):
        unique_id_dir[g] = cover_matrix_int[rind]

    unique_id_prob_dir = {}
    for z, t in unique_id_dir.items():
        bool_model = np.zeros(len(ruleset.data_info.target), dtype=bool)
        for i_tt, tt in enumerate(t):
            if tt == 1:
                if i_tt == len(ruleset.rules):
                    bool_model = ruleset.uncovered_bool
                else:
                    bool_model = np.bitwise_or(bool_model, ruleset.rules[i_tt].bool_array)
        unique_id_prob_dir[z] = calc_probs(ruleset.data_info.target[bool_model],
                                           ruleset.data_info.num_class)

    for i in range(len(prob_predicted)):
        prob_predicted[i] = unique_id_prob_dir[unique_id[i]]

    return prob_predicted
