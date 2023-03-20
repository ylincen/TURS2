import numpy as np


def get_readable_rule(rule):
    feature_names = rule.ruleset.data_info.feature_names
    readable = ""
    which_variables = np.where(rule.condition_count != 0)[0]
    for v in which_variables:
        cut = rule.condition_matrix[:, v][::-1]
        icol_name = feature_names[v]
        # readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + ";   "
        if np.isnan(cut[0]):
            cut_condition = "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
        elif np.isnan(cut[1]):
            cut_condition = "(X" + str(v) + ") " + icol_name + " >= " + str(cut[0]) + ";   "
        else:
            cut_condition = str(cut[0]) + " <=    " + "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
        readable += cut_condition
    readable = "If  " + readable

    readable += "\nProbability of READMISSION or NOT (in order): " + str(rule.prob_excl) + "\nNumber of patients who satisfy this rule: " + str(rule.coverage_excl) + "\n"
    return(readable)


def get_readable_rules(ruleset):
    readables = []
    for rule in ruleset.rules:
        readable = get_readable_rule(rule)
        readables.append(readable)
        print(readable)
    readable = "If none of above,\nProbability of READMISSION or NOT (in order): " + str(ruleset.else_rule_p) + "\nNumber of patients who do not satisfy any above rules: : " + str(ruleset.else_rule_coverage)
    print(readable)
    readables.append(readable)
