import numpy as np


def get_readable_rule(rule):
    feature_names = rule.ruleset.data_info.feature_names
    readable = ""
    which_variables = np.where(rule.condition_count != 0)[0]
    for v in which_variables:
        cut = rule.condition_matrix[:, v][::-1]
        icol_name = feature_names[v]
        readable += "X" + str(v) + "-" + icol_name + " in " + str(cut) + ";   "

    readable += "Prob: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)
    return(readable)


def get_readable_rules(ruleset, option="incl"):
    readables = []
    for rule in ruleset.rules:
        readable = ""
        which_variables = np.where(rule.condition_count != 0)[0]
        for i, v in enumerate(which_variables):
            cut = rule.condition_matrix[:, v][::-1]
            icol_name = ruleset.data_info.feature_names[v]
            if i == len(which_variables) - 1:
                readable += "X" + str(v) + "-" + str(icol_name) + " in " + str(cut) + "   ===>   "
            else:
                readable += "X" + str(v) + "-" + str(icol_name) + " in " + str(cut) + "   &   "

        if option == "incl":
            readable += "Prob Neg/Pos: " + str(rule.prob) + ", Coverage: " + str(rule.coverage)
        else:
            readable += "Prob Neg/Pos: " + str(rule.prob_excl) + ", Coverage: " + str(rule.coverage_excl)

        readables.append(readable)
        print(readable)

    readable = "Else-rule, Prob Neg/Pos: " + str(ruleset.else_rule_p) + ", Coverage: " + str(ruleset.else_rule_coverage)
    readables.append(readable)
    print(readable)