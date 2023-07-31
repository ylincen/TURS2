import numpy as np

def print_(rule):
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

    # readable += "\nProbability of NOT-READMISSION or READMISSION (in order): " + str(rule.prob) + "\nNumber of patients who satisfy this rule: " + str(rule.coverage) + "\n"
    if len(rule.prob) > 5:
        readable += "\nMax Prob: " + str(max(rule.prob)) + "; Number of patients who satisfy this rule: " + str(rule.coverage) + "\n"
    else:
        readable += "\nProbability: " + str(rule.prob) + "; Number of patients who satisfy this rule: " + str(rule.coverage) + "\n"

    return(readable)


def get_readable_rules(ruleset):
    readables = []
    for rule in ruleset.rules:
        readable = print_(rule)
        readables.append(readable)
        print(readable)
    readable = "If none of above,\nProbability of READMISSION or NOT (in order): " + str(ruleset.else_rule_p[::-1]) + "\nNumber of patients who do not satisfy any above rule: " + str(ruleset.else_rule_coverage)
    print(readable)
    readables.append(readable)
