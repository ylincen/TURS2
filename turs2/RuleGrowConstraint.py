# This script collects functions for constraining the rule grow process, incluidng
#   validity check (incl/excl), coverage reduction check, ...
import sys

from utils_calculating_cl import *


def validity_check(rule, icol, cut):
    res_excl = True
    res_incl = True
    if rule.data_info.alg_config.validity_check == "no_check":
        pass
    elif rule.data_info.alg_config.validity_check == "excl_check":
        res_excl = check_split_validity_excl(rule, icol, cut)
    elif rule.data_info.alg_config.validity_check == "incl_check":
        res_incl = check_split_validity(rule, icol, cut)
    elif rule.data_info.alg_config.validity_check == "either":
        res_excl = check_split_validity(rule, icol, cut)
        res_incl = check_split_validity_excl(rule, icol, cut)
    else:
        sys.exit("Error: the if-else statement should not end up here")
    return {"res_excl": res_excl, "res_incl": res_incl}

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

    validity = nll_rule + regret(rule.coverage, rule.data_info.num_class) - nll_left - nll_right - \
               regret(len(indices_left), rule.data_info.num_class) - \
               regret(len(indices_right), rule.data_info.num_class) - cl_model_extra

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

    validity = nll_rule + regret(rule.coverage_excl, rule.data_info.num_class) - nll_left - nll_right - regret(len(indices_left_excl), rule.data_info.num_class) - \
               regret(len(indices_right_excl), rule.data_info.num_class) - cl_model_extra

    validity_larger_than_zero = (validity > 0)

    return validity_larger_than_zero

def grow_cover_reduce_contraint(rule, nrow_data, nrow_data_excl, _coverage, _coverage_excl, alpha):
    dynamic_rate = (rule.coverage - _coverage) / rule.coverage
    dynamic_rate_excl = (rule.coverage_excl - _coverage_excl) / rule.coverage_excl

    static_rate = (rule.coverage - _coverage) / nrow_data
    static_rate_excl = (rule.coverage_excl - _coverage_excl) / nrow_data_excl

    _return = {"dynamic_ok": dynamic_rate <= alpha, "dynamic_excl_ok": dynamic_rate_excl <= alpha,
               "static_ok": static_rate <= alpha, "static_excl_ok": static_rate_excl <= alpha}
    return _return


