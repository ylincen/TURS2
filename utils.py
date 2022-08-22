from numba import njit
import numpy as np
import scipy.special
from constant import *

@njit
def calc_probs(target, num_class, smoothed=False):
    counts = np.bincount(target, minlength=num_class)
    if smoothed:
        counts = counts + np.ones(num_class, dtype='int64')
    if np.sum(counts) == 0:
        return counts / 1.0
    else:
        return counts / np.sum(counts)

@njit
def calc_shannofano_cl(target, smoothed=False):
    counts = np.bincount(target)
    if smoothed:
        counts = counts + np.ones(len(counts), dtype='int64')
    counts = counts[counts != 0]

    if np.sum(counts) == 0:
        p = counts / 1.0
    else:
        p = counts / np.sum(counts)
    return -np.sum(p * np.log2(p) * len(target))

@njit
def get_covered_indices_bool(unique_membership, membership):
    select_rows = np.ones(len(membership), dtype="bool")
    for row_i in range(len(membership) - 1):  # if row_i is a subset of row_j, or the reverse, exclude the bigger one
        if not unique_membership[row_i]:
            continue
        for row_j in range(row_i + 1, len(membership)):
            if not unique_membership[row_j]:
                continue
            if np.all(membership[row_j][membership[row_i]]):
                select_rows[row_j] = False  # exclude row_j
            elif np.all(membership[row_i][membership[row_j]]):
                select_rows[row_i] = False  # exclude row_i
            else:
                pass
    # covered_indices_bool = np.bitwise_or.reduce(membership[unique_membership & select_rows], axis=0)
    covered_indices_bool = (np.sum(membership[unique_membership & select_rows], axis=0) >= 1)
    return covered_indices_bool


def get_ruleset_fromchain_inclmodelcost(ruleset_chain, scores, incl_modelcost=False):
    if incl_modelcost:
        cl_model = []

        for r in ruleset_chain[-1].rules:
            cl_model.append(get_cl_model2(r))
        scores_incl_modelcost = scores + (np.append(0, np.cumsum(cl_model)) - (scipy.special.gammaln(np.arange(1, len(cl_model) + 2)) / np.log(2)))
        ruleset = ruleset_chain[np.argmin(scores_incl_modelcost)]
    else:
        ruleset = ruleset_chain[np.argmin(scores)]
    return ruleset

# get the most cost by considering that "the order of the literals does not matter", like in C4.5's book;
def get_cl_model2(rule):
    cl_model = []
    categorical_cuts = rule.get_categorical_cuts()
    covered_bool_sofar = np.ones(rule.data.n, dtype=bool)
    for icol in range(rule.data.ncol):
        if np.nansum(rule.cut_mat[:, icol]) == 0:
            continue
        else:
            if rule.data.var_types[icol] == NUMERIC:
                num_cuts = np.nansum(rule.cut_mat[:, icol])
                lower_bound, upper_bound = \
                    np.min(rule.data.X[covered_bool_sofar, icol]), np.max(rule.data.X[covered_bool_sofar, icol])
                num_candidate_cuts = np.sum((lower_bound < rule.data.candidate_cuts_search[icol]) &
                                            (rule.data.candidate_cuts_search[icol] < upper_bound))
                if num_candidate_cuts <= 1:
                    continue
                else:
                    if num_cuts == 1:
                        cl_model.append(np.log2(num_candidate_cuts))
                    else:
                        cl_model.append(np.log2(num_candidate_cuts * (num_candidate_cuts - 1) / 2))
            else:
                cl_model.append(rule.data.cached_model_cost[1][icol])
                covered_bool_here = np.isin(rule.data.X[:, icol], categorical_cuts[icol])
                covered_bool_sofar = covered_bool_sofar & covered_bool_here

    return max(np.sum(cl_model) - (scipy.special.gammaln(len(cl_model) + 1) / np.log(2)), 0)
