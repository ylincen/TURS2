import numpy as np
from utils_predict import *

from scipy.stats import chi2

def cover_matrix(ruleset, X):
    cover_matrix = np.zeros((len(X), len(ruleset.rules) + 1), dtype=bool)

    test_uncovered_bool = np.ones(len(X), dtype=bool)
    for ir, rule in enumerate(ruleset.rules):
        r_bool_array = np.ones(len(X), dtype=bool)

        condition_matrix = np.array(rule.condition_matrix)
        condition_count = np.array(rule.condition_count)
        which_vars = np.where(condition_count > 0)[0]

        upper_bound, lower_bound = condition_matrix[0], condition_matrix[1]
        upper_bound[np.isnan(upper_bound)] = np.Inf
        lower_bound[np.isnan(lower_bound)] = -np.Inf

        for v in which_vars:
            r_bool_array = r_bool_array & (X[:, v] < upper_bound[v]) & (X[:, v] >= lower_bound[v])

        cover_matrix[:, ir] = r_bool_array
        test_uncovered_bool = test_uncovered_bool & ~r_bool_array
    cover_matrix[:, -1] = test_uncovered_bool
    return cover_matrix

def predict_random_picking_for_overlaps(ruleset, X, seed):
    np.random.seed(seed)

    cover_mat = cover_matrix(ruleset, X)
    rules_probs = [r.prob for r in ruleset.rules] + [ruleset.else_rule_p]

    pred_probs = []
    for _ in cover_mat:
        rules_indices_ = np.where(_)[0]
        random_selected_ = np.random.randint(0, len(rules_indices_), size=1)[0]
        rule_index_selected = rules_indices_[random_selected_]

        pred_probs.append(rules_probs[rule_index_selected])

    return np.array(pred_probs, dtype=float)

def calculate_overlap_percentage(ruleset, X):
    cover_mat = cover_matrix(ruleset, X)
    num_rules_each_data = np.sum(cover_mat, axis=1) # sum each row
    return np.mean(num_rules_each_data > 1)
