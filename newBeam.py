import sys

import numpy as np
from newRule import *
import copy 
import surrogate_tree


class Beam:
    def __init__(self, beam_width, data_info):
        self.rules = []  # an array of Rules
        self.beam_width = beam_width
        self.data_info = data_info

    def grow_rule_incl(self, candidate_cuts):
        nml_foil_gain = []
        info_icol = []
        info_cut_index = []
        info_cut_type = []
        info_boolarray = []
        info_irules = []

        for irule, rule in enumerate(self.rules):
            for icol in range(rule.ncol):
                if rule.dim_type[icol] == NUMERIC:
                    candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features[:, icol])) & \
                                              (candidate_cuts[icol] > np.min(rule.features[:, icol]))
                    candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                    for i, cut in enumerate(candidate_cuts_icol):
                        # if cut == 0.1645 and icol == 6:
                        #     print("debug")
                        left_bi_array_incl = (rule.features[:, icol] < cut)
                        right_bi_array_incl = ~left_bi_array_incl

                        left_bi_array_excl = (rule.features_excl_overlap[:, icol] < cut)
                        right_bi_array_excl = ~left_bi_array_excl

                        left_local_score = rule.MDL_FOIL_gain(left_bi_array_excl, left_bi_array_incl, excl=False)
                        right_local_score = rule.MDL_FOIL_gain(right_bi_array_excl, right_bi_array_incl, excl=False)

                        if left_local_score > 0:
                            nml_foil_gain.append(left_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(LEFT_CUT)
                            info_boolarray.append(left_bi_array_incl)
                            info_irules.append(irule)
                        if right_local_score > 0:
                            nml_foil_gain.append(right_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(RIGHT_CUT)
                            info_boolarray.append(right_bi_array_incl)
                            info_irules.append(irule)
                else:
                    for i, level in enumerate(rule.categorical_levels[icol]):  # IMPLEMENT LATER
                        within_bi_array_incl = np.isin(rule.features[:, icol], level)
                        within_bi_array_excl = np.isin(rule.features_excl_overlap[:, icol], level)

                        within_local_score = rule.MDL_FOIL_gain(within_bi_array_excl, within_bi_array_incl,
                                                                excl=False)
                        if within_local_score > 0:
                            nml_foil_gain.append(within_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(WITHIN_CUT)
                            info_boolarray.append(within_bi_array_incl)
                            info_irules.append(irule)

        if len(nml_foil_gain) == 0:
            return []

        nml_foil_gain_sort_index = np.argsort(-np.array(nml_foil_gain))
        best_m_nmlfoilgain_index = []
        for kk, ind in enumerate(nml_foil_gain_sort_index):
            if len(best_m_nmlfoilgain_index) >= self.beam_width:
                break

            if kk == 0:
                best_m_nmlfoilgain_index.append(ind)
                continue

            diversity_skip = False
            for ll, best_ind in enumerate(best_m_nmlfoilgain_index):
                ind1 = self.rules[info_irules[ind]].indices[info_boolarray[ind]]
                ind2 = self.rules[info_irules[best_ind]].indices[info_boolarray[best_ind]]
                ind1_bool = np.zeros(self.data_info.nrow, dtype=bool)
                ind1_bool[ind1] = True
                ind2_bool = np.zeros(self.data_info.nrow, dtype=bool)
                ind2_bool[ind2] = True

                jarcard_dist = np.count_nonzero(np.bitwise_and(ind1_bool, ind2_bool)) / \
                               np.count_nonzero(np.bitwise_or(ind1_bool, ind2_bool))
                # jarcard_dist = np.count_nonzero(np.bitwise_or(info_boolarray[best_ind], info_boolarray[ind])) / \
                               # np.count_nonzero(np.bitwise_and(info_boolarray[best_ind], info_boolarray[ind]))
                # print("jarcard_dist: ", jarcard_dist)
                if jarcard_dist > 0.95:
                    diversity_skip = True
                    break

            if diversity_skip is False:
                best_m_nmlfoilgain_index.append(ind)
        best_rules = []

        for ind in best_m_nmlfoilgain_index:
            icol = info_icol[ind]
            cut_index = info_cut_index[ind]
            cut_type = info_cut_type[ind]
            rule = self.rules[info_irules[ind]]
            if cut_type == LEFT_CUT:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(rule.features[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                cut = candidate_cuts_icol[cut_index]

                bi_array_excl = (rule.features_excl_overlap[:, icol] < cut)
                bi_array_incl = (rule.features[:, icol] < cut)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [NUMERIC]
                cuts = cuts + [cut]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}
            elif cut_type == RIGHT_CUT:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(rule.features[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                cut = candidate_cuts_icol[cut_index]

                bi_array_excl = (rule.features_excl_overlap[:, icol] >= cut)
                bi_array_incl = (rule.features[:, icol] >= cut)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [NUMERIC]
                cuts = cuts + [cut]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}
            else:
                level = rule.categorical_levels[icol][cut_index]

                bi_array_excl = np.isin(rule.features_excl_overlap[:, icol], level)
                bi_array_incl = np.isin(rule.features[:, icol], level)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [CATEGORICAL]
                cuts = cuts + [level]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}

            rule = Rule(indices=rule.indices[bi_array_incl],
                        indices_excl_overlap=rule.indices_excl_overlap[bi_array_excl],
                        rule_base=rule,
                        features=rule.features[bi_array_incl],
                        target=rule.target[bi_array_incl],
                        features_excl_overlap=rule.features_excl_overlap[bi_array_excl],
                        target_excl_overlap=rule.target_excl_overlap[bi_array_excl],
                        data_info=rule.data_info,
                        condition=condition,
                        local_gain=nml_foil_gain[ind])

            best_rules.append(rule)

        return best_rules


    def grow_rule_excl(self, candidate_cuts):
        """
        Grow the rule by adding one literal to it, while ignoring/excluding all covered instances in the ruleset
        :param beam: an object from class Beam (newBeam.py): a new beam that contains all rules with one more rule than
        this Rule.
        :param candidate_cuts: a dictionary, to store the candidate cuts based on the ORIGINAL FULL DATASET
        :return: a beam of rules
        """
        nml_foil_gain = []
        info_icol = []
        info_cut_index = []
        info_cut_type = []
        info_boolarray = []
        info_irules = []
        for irule, rule in enumerate(self.rules):
            for icol in range(rule.ncol):
                if rule.dim_type[icol] == NUMERIC:
                    # constrain the search space
                    candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features_excl_overlap[:, icol])) & \
                                              (candidate_cuts[icol] >= np.min(rule.features_excl_overlap[:, icol]))
                    candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]

                    # generate & evaluate all possible growth
                    for i, cut in enumerate(candidate_cuts_icol):
                        left_bi_array = (rule.features_excl_overlap[:, icol] < cut)
                        right_bi_array = ~left_bi_array

                        left_local_score = rule.MDL_FOIL_gain(bi_array_excl=left_bi_array)  # IMPLEMENT LATER
                        right_local_score = rule.MDL_FOIL_gain(bi_array_excl=right_bi_array)

                        if left_local_score > 0:
                            nml_foil_gain.append(left_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(LEFT_CUT)
                            info_boolarray.append(left_bi_array)
                            info_irules.append(irule)
                        if right_local_score > 0:
                            nml_foil_gain.append(right_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(RIGHT_CUT)
                            info_boolarray.append(right_bi_array)
                            info_irules.append(irule)
                else:
                    for i, level in enumerate(rule.categorical_levels[icol]):
                        within_bi_array = np.isin(rule.features_excl_overlap[:, icol], level)
                        within_local_score = rule.MDL_FOIL_gain(bi_array_excl=within_bi_array)
                        if within_local_score > 0:
                            nml_foil_gain.append(within_local_score)
                            info_icol.append(icol)
                            info_cut_index.append(i)
                            info_cut_type.append(WITHIN_CUT)
                            info_boolarray.append(within_bi_array)
                            info_irules.append(irule)

        if len(nml_foil_gain) == 0:
            return []

        nml_foil_gain_sort_index = np.argsort(-np.array(nml_foil_gain))
        best_m_nmlfoilgain_index = []
        for kk, ind in enumerate(nml_foil_gain_sort_index):
            if len(best_m_nmlfoilgain_index) >= self.beam_width:
                break

            if kk == 0:
                best_m_nmlfoilgain_index.append(ind)
                continue

            diversity_skip = False
            for ll, best_ind in enumerate(best_m_nmlfoilgain_index):
                # jarcard_dist = np.count_nonzero(np.bitwise_and(info_boolarray[best_ind], info_boolarray[ind])) / \
                #                np.count_nonzero(np.bitwise_or(info_boolarray[best_ind], info_boolarray[ind]))
                ind1 = self.rules[info_irules[ind]].indices_excl_overlap[info_boolarray[ind]]
                ind2 = self.rules[info_irules[best_ind]].indices_excl_overlap[info_boolarray[best_ind]]
                ind1_bool = np.zeros(self.data_info.nrow, dtype=bool)
                ind1_bool[ind1] = True
                ind2_bool = np.zeros(self.data_info.nrow, dtype=bool)
                ind2_bool[ind2] = True

                jarcard_dist = np.count_nonzero(np.bitwise_and(ind1_bool, ind2_bool)) / \
                               np.count_nonzero(np.bitwise_or(ind1_bool, ind2_bool))
                if jarcard_dist > 0.95:
                    diversity_skip = True
                    break
            if diversity_skip is False:
                best_m_nmlfoilgain_index.append(ind)

        best_rules = []
        for ind in best_m_nmlfoilgain_index:
            icol = info_icol[ind]
            cut_index = info_cut_index[ind]
            cut_type = info_cut_type[ind]
            rule = self.rules[info_irules[ind]]
            if cut_type == LEFT_CUT:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features_excl_overlap[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(rule.features_excl_overlap[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                cut = candidate_cuts_icol[cut_index]

                bi_array_excl = (rule.features_excl_overlap[:, icol] < cut)
                bi_array_incl = (rule.features[:, icol] < cut)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [NUMERIC]
                cuts = cuts + [cut]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}
            elif cut_type == RIGHT_CUT:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(rule.features_excl_overlap[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(rule.features_excl_overlap[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                cut = candidate_cuts_icol[cut_index]

                bi_array_excl = (rule.features_excl_overlap[:, icol] >= cut)
                bi_array_incl = (rule.features[:, icol] >= cut)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [NUMERIC]
                cuts = cuts + [cut]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}
            else:
                level = rule.categorical_levels[icol][cut_index]

                bi_array_excl = np.isin(rule.features_excl_overlap[:, icol], level)
                bi_array_incl = np.isin(rule.features[:, icol], level)

                icols, var_types, cuts, cut_options = list(rule.condition.values())
                icols = icols + [icol]
                var_types = var_types + [CATEGORICAL]
                cuts = cuts + [level]
                cut_options = cut_options + [cut_type]
                condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                             "cut_options": cut_options}

            rule = Rule(indices=rule.indices[bi_array_incl],
                        indices_excl_overlap=rule.indices_excl_overlap[bi_array_excl],
                        rule_base=rule,
                        features=rule.features[bi_array_incl],
                        target=rule.target[bi_array_incl],
                        features_excl_overlap=rule.features_excl_overlap[bi_array_excl],
                        target_excl_overlap=rule.target_excl_overlap[bi_array_excl],
                        data_info=rule.data_info,
                        condition=condition,
                        local_gain=nml_foil_gain[ind])

            best_rules.append(rule)

        return best_rules


class BeamCollect:
    def __init__(self, beam_width, beams, similarity_alpha):
        self.beam_width = beam_width
        self.similarity_alpha = similarity_alpha

        self.beams = beams
        self.rules = self.flat_beams()
    
    def flat_beams(self):
        rules = []
        for beam in self.beams:
            for rule in beam:
                rules.append(rule)
        return rules

    def select_best_m(self, else_rule, ruleset, m=None):
        if m is None:
            m = self.beam_width
        surrogate_scores = []
        for rule in self.rules:
            rule_score = rule.neglog_likelihood_excl + rule.regret_excl
        
            else_rule_cover_updated = copy.deepcopy(else_rule.bool_array)
            else_rule_cover_updated[rule.indices_excl_overlap] = False
        
            if any(else_rule_cover_updated):
                surrogate_score_else_rule = \
                    surrogate_tree.get_tree_cl(x_train=ruleset.features[else_rule_cover_updated],
                                               y_train=ruleset.target[else_rule_cover_updated],
                                               num_class=ruleset.data_info.num_class)
            else:
                surrogate_score_else_rule = 0
            surrogate_score = rule_score + np.sum(surrogate_score_else_rule)
            surrogate_scores.append(surrogate_score)

        best_m_rules_are = np.argsort(surrogate_scores)
        rules_with_diversity_constraint = []
        for which_rule in best_m_rules_are:
            if len(rules_with_diversity_constraint) == 0:
                rules_with_diversity_constraint.append(self.rules[which_rule])
            elif len(rules_with_diversity_constraint) >= m:
                return rules_with_diversity_constraint
            elif which_rule == best_m_rules_are[-1]:
                return rules_with_diversity_constraint
            else:
                # diversity check
                bool_array_this = self.rules[which_rule].bool_array_excl
                for j, r in enumerate(rules_with_diversity_constraint):
                    cover_overlap_count = np.count_nonzero(np.bitwise_and(self.rules[j].bool_array_excl, bool_array_this))
                    cover_union_count = np.count_nonzero(np.bitwise_or(self.rules[j].bool_array_excl, bool_array_this))
                    if cover_overlap_count / cover_union_count > self.similarity_alpha:
                        break
                else:
                    rules_with_diversity_constraint.append(self.rules[which_rule])