# This script collect functions for the experiment section of the paper
# Specifically, 1) explainability, 2) Truly Unordered Verfitication; 3) Significance of rules; 4) Insignificance of overlaps

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

def explainability_analysis(ruleset, X):
    cover_mat_NoElseRule = cover_matrix(ruleset, X)[:, :-1]
    literal_length_mat = np.zeros((len(X), len(ruleset.rules)), dtype=float)

    rule_lengths_ = np.array([np.sum(r.condition_count) for r in ruleset.rules], dtype=float)

    for irule in range(len(ruleset.rules)):
        literal_length_mat[cover_mat_NoElseRule[:, irule], irule] = rule_lengths_[irule]

    num_literals_each_data = np.sum(literal_length_mat, axis=1) # sum each row
    return np.mean(num_literals_each_data[num_literals_each_data != 0])  # exclude data points covered by else-rule;

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

def significance_rules(ruleset, X_test, y_test, num_permutations):
    p_values_chisquare = []
    p_values_permutations = []

    rules_local_prediction = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rules_prob_test_data = rules_local_prediction["rules_test_p"]
    rules_test_p_NotThisRule_including_elserule = rules_local_prediction["rules_test_p_NotThisRule_including_elserule"]
    rules_test_bool_array = rules_local_prediction["rules_cover_test_including_elserule"]

    def calculate_reduced_model_loglikelihood():
        p_default_testdata = calc_probs(y_test, ruleset.data_info.num_class)
        p_default_testdata = p_default_testdata[p_default_testdata != 0]
        reduced_model_loglikelihood = np.sum(np.log(p_default_testdata) * p_default_testdata * len(y_test))
        return reduced_model_loglikelihood

    def calculated_chi2_pvalue(stat):
        return 1 - chi2(1).cdf(stat)

    def calculate_full_model_loglikelihood():
        p_rule = rules_prob_test_data[ir]
        p_notThisRule = rules_test_p_NotThisRule_including_elserule[ir]
        full_model_loglikelihood = np.sum(np.log(p_rule[y_test[rules_test_bool_array[ir]]])) + \
                                   np.sum(np.log(p_notThisRule[y_test[~rules_test_bool_array[ir]]]))
        return full_model_loglikelihood

    def calculated_permutation_pvalue(reduced_model_loglikelihood):
        np.random.seed(1)
        permu_counter = 0
        for iter in range(num_permutations):
            y_permutation = np.random.permutation(y_test)
            p_permu = calc_probs(y_permutation[rules_test_bool_array[ir]], ruleset.data_info.num_class)
            p_permu_NotThisRule = calc_probs(y_permutation[~rules_test_bool_array[ir]], ruleset.data_info.num_class)
            permu_full_model_loglikelihood = \
                np.sum(np.log(p_permu[y_permutation[rules_test_bool_array[ir]]])) + \
                np.sum(np.log(p_permu_NotThisRule[y_permutation[~rules_test_bool_array[ir]]]))
            permu_test_stat = -2 * (reduced_model_loglikelihood - permu_full_model_loglikelihood)
            permu_counter += (test_stat <= permu_test_stat)
        return permu_counter / num_permutations

    for ir, rule in enumerate(ruleset.rules):
        # NOTE: use np.log(..) instead of np.log2(..) here;
        reduced_model_loglikelihood = calculate_reduced_model_loglikelihood()
        full_model_loglikelihood = calculate_full_model_loglikelihood()
        test_stat = -2 * (reduced_model_loglikelihood - full_model_loglikelihood)

        p_values_chisquare.append(calculated_chi2_pvalue(test_stat))
        p_values_permutations.append(calculated_permutation_pvalue(reduced_model_loglikelihood))

    return [p_values_chisquare, p_values_permutations]

def insignificance_overlap(ruleset, mg, X_test, y_test, num_permutation):
    rules_local_prediction = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rules_test_bool_array = rules_local_prediction["rules_cover_test_including_elserule"]
    rules_prob_test_data = rules_local_prediction["rules_test_p"]
    def calculate_reduced_model_loglikelihood(rule_index):
        # note that "reduced" means the NULL hypothesis;
        p_rule = rules_prob_test_data[rule_index]
        coverage = np.count_nonzero(rules_test_bool_array[rule_index])
        return np.sum(np.log(p_rule) * p_rule * coverage)

    def calculate_full_model_loglikelihood(intersection_cover, remaining_cover, y_):
        # y_ can be y_test or y_permu;
        def part_log_likelihood_(cover):
            p_cover = calc_probs(y_[cover], ruleset.data_info.num_class)
            p_cover = p_cover[p_cover != 0]
            coverage = np.count_nonzero(cover)
            return np.sum(np.log(p_cover) * p_cover * coverage)

        return part_log_likelihood_(intersection_cover) + part_log_likelihood_(remaining_cover)

    def calculate_permutation_pvalue(num_permutation, union_cover, intersection_cover):
        np.random.seed(1)
        counter = 0
        for iter in range(num_permutation):
            y_permu = np.array(y_test)
            y_permu[union_cover] = np.random.permutation(y_test[union_cover])

            remaining_cover = union_cover & (~intersection_cover)
            permu_full_model_loglikelihood = \
                calculate_full_model_loglikelihood(intersection_cover=intersection_cover,
                                                   remaining_cover=remaining_cover, y_=y_permu)
            permu_stat = -2 * (reduced_model_loglikelihood - permu_full_model_loglikelihood)
            if permu_stat >= stat:  # if stat is very "extreme", it's unlikely permu_stat will be even larger.
                counter += 1
        return counter / num_permutation

    intersection_cover = np.ones(len(y_test), dtype=bool)
    union_cover = np.zeros(len(y_test), dtype=bool)
    for index_rule in mg.rules_involvde:
        intersection_cover = intersection_cover & rules_test_bool_array[index_rule]
        union_cover = union_cover | rules_test_bool_array[index_rule]

    if np.count_nonzero(intersection_cover) == 0:
        return [np.nan]

    full_model_loglikelihood = \
        calculate_full_model_loglikelihood(intersection_cover=intersection_cover,
                                           remaining_cover=union_cover & (~intersection_cover),
                                           y_=y_test)
    union_p = calc_probs(y_test[union_cover], ruleset.data_info.num_class)
    union_p = union_p[union_p != 0]
    reduced_model_loglikelihood = sum(np.log(union_p) * union_p * np.count_nonzero(union_cover))
    stat = -2 * (reduced_model_loglikelihood - full_model_loglikelihood)
    p_value = calculate_permutation_pvalue(num_permutation=num_permutation,
                                           union_cover=union_cover,
                                           intersection_cover=intersection_cover)

    return [p_value]

def insignificance_overlap_all(ruleset, X_test, y_test, num_permutation):
    p_values_all = {}
    p_values_all_flatten = []
    for i_mg, mg in enumerate(ruleset.modelling_groups):
        if len(mg.rules_involvde) > 1:
            p_values = insignificance_overlap(ruleset, mg, X_test, y_test, num_permutation)
            p_values_all[i_mg] = p_values
            p_values_all_flatten.extend(p_values)

    p_values_all_flatten = np.array(p_values_all_flatten)
    p_values_all_flatten = p_values_all_flatten[~np.isnan(p_values_all_flatten)]
    insig_perc = np.mean(p_values_all_flatten > 0.05)
    return {"p_values_all":p_values_all, "insignificance_percentage": insig_perc}

