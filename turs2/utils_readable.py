import numpy as np

def print_(rule, feature_names_=None, round_ = 3, class_names=None):
    if feature_names_ is None:
        feature_names = rule.ruleset.data_info.feature_names
    else:
        feature_names = feature_names_
    readable = ""
    which_variables = np.where(rule.condition_count != 0)[0]
    for v in which_variables:
        cut = rule.condition_matrix[:, v][::-1]
        icol_name = str(feature_names[v])
        if np.isnan(cut[0]):
            if cut[1] == 0.5 and len(rule.data_info.candidate_cuts[v]) == 1:
                # cut_condition = "(X" + str(v) + "[binary variable]) " + icol_name + " = " + "0" + ";   "
                cut_condition = icol_name + " = " + "0" + ";   "
            else:
                cut[1] = np.round(cut[1], round_)
                # cut_condition = "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
                cut_condition = icol_name + " < " + str(cut[1]) + ";   "
        elif np.isnan(cut[1]):
            if cut[0] == 0.5 and len(rule.data_info.candidate_cuts[v]) == 1:
                # cut_condition = "(X" + str(v) + "[binary variable]) " + icol_name + " = " + "1" + ";   "
                cut_condition = icol_name + " = " + "1" + ";   "
            else:
                cut[0] = np.round(cut[0], round_)
                # cut_condition = "(X" + str(v) + ") " + icol_name + " >= " + str(cut[0]) + ";   "
                cut_condition = icol_name + " >= " + str(cut[0]) + ";   "
        else:
            cut_condition = str(cut[0]) + " <=    " + "(X" + str(v) + ") " + icol_name + " < " + str(cut[1]) + ";   "
        readable += cut_condition
    readable = "If  " + readable

    # readable += "\nProbability of NOT-READMISSION or READMISSION (in order): " + str(rule.prob) + "\nNumber of patients who satisfy this rule: " + str(rule.coverage) + "\n"
    if class_names is None:
        if len(rule.prob) > 20:
            readable += "\nMax Prob: " + str(max(rule.prob)) + "; Coverage: " + str(rule.coverage) + "\n"
        else:
            readable += "\nProbability: " + str(np.round(rule.prob, round_)) + "; Coverage: " + str(rule.coverage) + "\n"
    else:
        for i_, c_name in enumerate(class_names):
            if i_ == 0:
                readable += "\nProbability of " + c_name + ": " + str(np.round(rule.prob[i_], round_)) + "\n"
            elif i_ == len(class_names) - 1:
                readable += "Probability of " + c_name + ": " + str(np.round(rule.prob[i_], round_)) + ";\n Number of data points (rows) that satisfy this rule's condition: " + str(rule.coverage) + "\n"
            else:
                readable += "Probability of " + c_name + ": " + str(np.round(rule.prob[i_], round_)) + "\n"
    return(readable)


def get_readable_rules(ruleset):
    readables = []
    for rule in ruleset.rules:
        readable = print_(rule)
        readables.append(readable)
        print(readable)
    readable = "If none of above,\nProbability: " + str(ruleset.else_rule_p[::-1]) + "\nCoverage: " + str(ruleset.else_rule_coverage)
    print(readable)
    readables.append(readable)
