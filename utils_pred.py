import numpy as np

from constant import *
from utils import *


def cover_by_this_rule(rule, x):
    """
    Test whether "rule" covers "x"
    """
    icols, var_types, cuts, cuts_options = rule.condition.values()
    for i, icol in enumerate(icols):
        if cuts_options[i] == LEFT_CUT:
            if x[icol] < cuts[i]:
                pass
            else:
                return False
        elif cuts_options[i] == RIGHT_CUT:
            if x[icol] >= cuts[i]:
                pass
            else:
                return False
        else:
            if np.isin(x[icol], cuts[i]):
                pass
            else:
                return False
    else:
        return True


def test_bool(rule, X):
    """
    return a boolean array representing which rows in the X_test that "rule" covers
    """
    return [cover_by_this_rule(rule, x) for x in X]


def get_test_p(ruleset, X):
    """
    return the predicted probs of X_test
    """
    test_bool_all = []
    for rule in ruleset.rules:
        test_bool_all.append(test_bool(rule, X))

    test_p = np.zeros((len(X), ruleset.data_info.num_class), dtype=float)
    test_bool_all = np.array(test_bool_all, dtype=bool)

    if len(ruleset.rules) == 0:
        test_p = np.repeat(ruleset.else_rule.p, len(X)).reshape((ruleset.data_info.num_class, len(X))).T
    else:
        for icol in range(test_bool_all.shape[1]):
            rules_involved = test_bool_all[:, icol]
            if np.count_nonzero(rules_involved) == 0:
                test_p[icol] = ruleset.modeling_groups.else_rule_modeling_group.p
            else:
                for mdg in ruleset.modeling_groups.modeling_group_set:
                    if np.all(mdg.rules_involved_boolean == rules_involved):
                        test_p[icol] = mdg.p
                        break
                else:
                    modelling_bool = np.zeros(ruleset.data_info.nrow, dtype=bool)
                    for index in np.where(rules_involved)[0]:
                        modelling_bool = np.bitwise_or(modelling_bool, ruleset.rules[index].bool_array)
                    test_p[icol] = calc_probs(ruleset.data_info.target[modelling_bool], ruleset.data_info.num_class)
    return test_p



