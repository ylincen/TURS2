# This script collect functions for the experiment section of the paper
# Specifically, 1) explainability, 2) Truly Unordered Verfitication; 3) Significance of rules; 4) Insignificance of overlaps

import numpy as np

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

def explainability_analysis(ruleset, X):
    cover_mat_NoElseRule = cover_matrix(ruleset, X)[:, :-1]
    literal_length_mat = np.zeros((len(X), len(ruleset.rules)), dtype=float)

    rule_lengths_ = np.array([np.sum(r.condition_count) for r in ruleset.rules], dtype=float)

    for irule in range(len(ruleset.rules)):
        literal_length_mat[cover_mat_NoElseRule[:, irule], irule] = rule_lengths_[irule]

    num_literals_each_data = np.sum(literal_length_mat, axis=1) # sum each row
    return np.mean(num_literals_each_data[num_literals_each_data != 0])  # exclude data points covered by else-rule;

def predict_random_picking_for_overlaps(ruleset, X):
    cover_mat = cover_matrix(ruleset, X)
    rules_probs = [r.prob for r in ruleset.rules]








