import numpy as np
import pandas as pd

from exp_predictive_perf import *


def print_rule(rule):
    feature_names = rule.ruleset.data_info.feature_names
    readable = ""
    which_variables = np.where(rule.condition_count != 0)[0]
    for v in which_variables:
        cut = rule.condition_matrix[:, v][::-1]
        icol_name = str(feature_names[v])
        if np.isnan(cut[0]):
            if cut[1] == 0.5 and len(rule.data_info.candidate_cuts[v]) == 1:
                cut_condition = "(X" + str(v) + "[binary variable]) " + icol_name + " = " + "0" + ";   "
            else:
                cut_condition = "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
        elif np.isnan(cut[1]):
            if cut[0] == 0.5 and len(rule.data_info.candidate_cuts[v]) == 1:
                cut_condition = "(X" + str(v) + "[binary variable]) " + icol_name + " = " + "1" + ";   "
            else:
                cut_condition = "(X" + str(v) + ") " + icol_name + " >= " + str(cut[0]) + ";   "
        else:
            cut_condition = str(cut[0]) + " <=    " + "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
        readable += cut_condition
    readable = "If  " + readable
    readable += "\nProbability of READMISSION: " + str(rule.prob[1]) + "\nNumber of patients who satisfy this rule: " + str(rule.coverage) + "\n"
    print(readable)


def print_ruleset(ruleset):
    for r in ruleset.rules:
        print_rule(r)

    readable = "If none of above,\nProbability of READMISSION: " + str(
        ruleset.else_rule_p[1]) + "\nNumber of patients who do not satisfy any above rule: " + str(
        ruleset.else_rule_coverage)
    print(readable)


def rules_quality(ruleset, X_test, y_test, X_train, y_train):
    X_test = X_test.to_numpy()
    X_train = X_train.to_numpy()

    y_train, y_test = y_train.to_numpy().ravel(), y_test.to_numpy().ravel()
    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, "ICU", 0, 0, 0)

    sig_res = exp_res["rules_p_value_permutations"]
    train_probs = [r.prob for r in ruleset.rules]


    local_res = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    test_probs = local_res["rules_test_p"]

    return pd.DataFrame({
        "readmission prob train": [np.round(p[1], 4) for p in train_probs],
        "readmission prob test": [np.round(p[1], 4) for p in test_probs],
        "permutation p value on test": np.round(sig_res, 4)
    })


