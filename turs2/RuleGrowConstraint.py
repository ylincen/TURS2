# This script collects functions for constraining the rule grow process, incluidng
#   validity check (incl/excl), coverage reduction check, ...
import sys

from utils_calculating_cl import *


def validity_check(rule, icol, cut):
    if rule.data_info.alg_config.validity_check == "no_check":
        res = True
    elif rule.data_info.alg_config.validity_check == "excl_check":
        res = check_split_validity_excl(rule, icol, cut)
    elif rule.data_info.alg_config.validity_check == "incl_check":
        res = check_split_validity(rule, icol, cut)
    else:
        sys.exit("Error: the if-else statement should not end up here")
    return res

def check_split_validity(rule, icol, cut):
    indices_left, indices_right = rule.indices[rule.features[:, icol] < cut], rule.indices[rule.features[:, icol] >= cut]

    p_rule = rule.prob
    p_left = calc_probs(rule.data_info.target[indices_left], rule.data_info.num_class)
    p_right = calc_probs(rule.data_info.target[indices_right], rule.data_info.num_class)

    nll_rule = calc_negloglike(p_rule, rule.coverage)
    nll_left = calc_negloglike(p_left, len(indices_left))
    nll_right = calc_negloglike(p_right, len(indices_right))

    cl_model_extra = rule.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
    cl_model_extra += np.log2(rule.ruleset.model_encoding.data_ncol_for_encoding)
    num_vars = np.sum(rule.condition_count > 0)
    cl_model_extra += rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                      rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

    validity = nll_rule + regret(rule.coverage, 2) - nll_left - nll_right - regret(len(indices_left), 2) - regret(len(indices_right), 2) - cl_model_extra

    validity_larger_than_zero = (validity > 0)

    return validity_larger_than_zero

def check_split_validity_excl(rule, icol, cut):
    indices_left_excl, indices_right_excl = rule.indices_excl_overlap[rule.features_excl_overlap[:, icol] < cut], \
        rule.indices_excl_overlap[rule.features_excl_overlap[:, icol] >= cut]

    p_rule = rule.prob_excl
    p_left = calc_probs(rule.data_info.target[indices_left_excl], rule.data_info.num_class)
    p_right = calc_probs(rule.data_info.target[indices_right_excl], rule.data_info.num_class)

    nll_rule = calc_negloglike(p_rule, rule.coverage_excl)
    nll_left = calc_negloglike(p_left, len(indices_left_excl))
    nll_right = calc_negloglike(p_right, len(indices_right_excl))

    cl_model_extra = rule.ruleset.model_encoding.cached_cl_model["l_cut"][0][icol]  # 0 represents for the "one split cut", instead of "two-splits cut"
    cl_model_extra += np.log2(rule.ruleset.model_encoding.data_ncol_for_encoding)
    num_vars = np.sum(rule.condition_count > 0)
    cl_model_extra += rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars + 1] - \
                      rule.ruleset.model_encoding.cached_cl_model["l_number_of_variables"][num_vars]

    validity = nll_rule + regret(rule.coverage_excl, 2) - nll_left - nll_right - regret(len(indices_left_excl), 2) - \
               regret(len(indices_right_excl), 2) - cl_model_extra

    validity_larger_than_zero = (validity > 0)

    return validity_larger_than_zero

def grow_cover_reduce_contraint(rule, data_info, _coverage, _coverage_excl, alpha):
    N = data_info.nrow
    dynamic_rate = (rule.coverage - _coverage) / rule.coverage
    dynamic_rate_excl = (rule.coverage_excl - _coverage_excl) / rule.coverage_excl

    static_rate = (rule.coverage - _coverage) / N
    static_rate_excl = (rule.coverage_excl - _coverage_excl) / N

    _return = {"dynamic_ok": dynamic_rate <= alpha, "dynamic_excl_ok": dynamic_rate_excl <= alpha,
               "static_ok": static_rate <= alpha, "static_excl_ok": static_rate_excl <= alpha}
    return _return


