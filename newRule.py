from constant import *
import numpy as np
from newBeam import *
from utils import *
from nml_regret import *
import surrogate_tree
import itertools


class Rule:
    def __init__(self, indices, indices_excl_overlap, rule_base, features, target,
                 features_excl_overlap, target_excl_overlap, data_info, condition,
                 local_gain):
        """
        :param indices: indices in the original full dataset that this rule covers
        :param rule_base: rule_base that is used to obtain this rule
        :param features: a numpy 2d array, representing the feature matrix (that is covered by this rule)
        :param target: a numpy 1d array, representing the class labels (that is covered by this rule)
        :param data_info: meta-data for the ORIGINAL WHOLE DATASET, including:
                    - type of each dimension, NUMERIC or CATEGORICAL
                    - num_class: number of class in the target variable;
                    - nrow: number of rows in the original dataset;
        :param condition:
        :return:
        """
        self.indices = indices  # corresponds to the original dataset
        self.indices_excl_overlap = indices_excl_overlap  # corresponds to the original
        self.data_info = data_info  # meta data of the original whole dataset
        self.dim_type = data_info.dim_type

        self.bool_array = self.get_bool_array(self.indices)
        self.bool_array_excl = self.get_bool_array(self.indices_excl_overlap)

        self.rule_base = rule_base  # the previous level of this rule, i.e., the rule from which "self" is obtained

        self.features = features  # feature rows covered by this rule
        self.target = target  # target sub-vector covered by this rule
        self.features_excl_overlap = features_excl_overlap  # feature rows covered by this rule WITHOUT ruleset's cover
        self.target_excl_overlap = target_excl_overlap  # target covered by this rule WITHOUT ruleset's cover

        self.nrow, self.ncol = features.shape  # local nrow and local ncol
        self.nrow_excl, self.ncol_excl = features_excl_overlap.shape  # local nrow and ncol, excluding ruleset's cover

        self.condition = condition  # condition is a dictionary with keys {icols, var_types, cuts, cut_options}

        self.categorical_levels = self.get_categorical_levels(max_number_levels_together=5)

        self.prob_excl = self._calc_prob_excl()
        self.prob = self._calc_prob()
        self.regret_excl = self._regret(self.nrow_excl, data_info.num_class)
        self.regret = self._regret(self.nrow, data_info.num_class)

        # code length of encoding nrow_excl instances by prob_excl
        p_selector = (self.prob_excl != 0)
        self.neglog_likelihood_excl = \
            -np.sum(self.prob_excl[p_selector] *
                    np.log2(self.prob_excl[p_selector])) * self.nrow_excl

        # code length of encoding nrow_excl instances by prob
        p_selector2 = (self.prob != 0)
        self.neglog_likelihood_incl = -np.sum(self.prob_excl[p_selector2] *
                                              np.log2(self.prob[p_selector2])) * self.nrow_excl

        self.local_gain = local_gain

    def get_categorical_levels(self, max_number_levels_together):
        categorical_levels = {}
        for icol in range(self.ncol):
            if self.data_info.dim_type[icol] == CATEGORICAL:
                unique_feature = np.unique(self.features[:,icol])
                candidate_cut_this_dimension = []

                if max_number_levels_together < len(unique_feature):
                    for i in range(max_number_levels_together):
                        candidate_cut_this_dimension.extend(list(itertools.combinations(unique_feature, r=i + 1)))
                else:
                    for i in range(len(unique_feature) - 1):
                        candidate_cut_this_dimension.extend(list(itertools.combinations(unique_feature, r=i + 1)))

                categorical_levels[icol] = candidate_cut_this_dimension

        return categorical_levels

    def get_bool_array(self, indices):
        bool_array = np.zeros(self.data_info.nrow, dtype=bool)
        bool_array[indices] = True
        return bool_array

    def grow_incl(self, candidate_cuts, beam):
        """
        Grow "self" by first generating possible growth and compare the growth with rules in the beam, and
        update the beam accordingly (if the growth is better than the worst one in the beam)
        :param candidate_cuts:
        :param beam:
        :return:
        """
        for icol in range(self.ncol):
            if self.dim_type[icol] == NUMERIC:
                candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features[:, icol])) & \
                                          (candidate_cuts[icol] > np.min(self.features[:, icol]))
                candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
                for i, cut in enumerate(candidate_cuts_icol):
                    left_bi_array_incl = (self.features[:, icol] < cut)
                    right_bi_array_incl = ~left_bi_array_incl

                    left_bi_array_excl = (self.features_excl_overlap[:, icol] < cut)
                    right_bi_array_excl = ~left_bi_array_excl

                    left_local_score = self.MDL_FOIL_gain(left_bi_array_excl, left_bi_array_incl, excl=False)
                    right_local_score = self.MDL_FOIL_gain(right_bi_array_excl, right_bi_array_incl, excl=False)

                    # check whether the beam should be updated
                    if left_local_score > right_local_score:
                        beam.update(rule_base=self, local_gain=left_local_score, bi_array_excl=left_bi_array_excl,
                                    icol=icol, var_type=NUMERIC, cut_type=LEFT_CUT, cut=cut,
                                    excl_or_not=False, bi_array_incl=left_bi_array_incl, buffer=None)
                    else:
                        beam.update(rule_base=self, local_gain=right_local_score, bi_array_excl=right_bi_array_excl,
                                    icol=icol, var_type=NUMERIC, cut_type=RIGHT_CUT, cut=cut,
                                    excl_or_not=False, bi_array_incl=right_bi_array_incl, buffer=None)

            else:
                for i, level in enumerate(self.categorical_levels[icol]):  # IMPLEMENT LATER
                    within_bi_array_incl = np.isin(self.features[:, icol], level)
                    within_bi_array_excl = np.isin(self.features_excl_overlap[:, icol], level)

                    within_local_score = self.MDL_FOIL_gain(within_bi_array_excl, within_bi_array_incl, excl=False)

                    beam.update(rule_base=self, local_gain=within_local_score, bi_array_excl=within_bi_array_excl,
                                icol=icol, var_type=CATEGORICAL, cut_type=WITHIN_CUT, cut=level,
                                excl_or_not=False, bi_array_incl=within_bi_array_incl, buffer=None)
        return beam

    # def grow_excl(self, candidate_cuts, beam):
    #     """
    #     Grow the rule by adding one literal to it, while ignoring/excluding all covered instances in the ruleset
    #     :param beam: an object from class Beam (newBeam.py): a new beam that contains all rules with one more rule than
    #     this Rule.
    #     :param candidate_cuts: a dictionary, to store the candidate cuts based on the ORIGINAL FULL DATASET
    #     :return: a beam of rules
    #     """
    #     nml_foil_gain = []
    #     info_icol = []
    #     info_cut_index = []
    #     info_cut_type = []
    #     info_boolarray = []
    #     for icol in range(self.ncol):
    #         if self.dim_type[icol] == NUMERIC:
    #             # constrain the search space
    #             candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    #                                       (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
    #             candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #
    #             # generate & evaluate all possible growth
    #             for i, cut in enumerate(candidate_cuts_icol):
    #                 left_bi_array = (self.features_excl_overlap[:, icol] < cut)
    #                 right_bi_array = ~left_bi_array
    #
    #                 left_local_score = self.MDL_FOIL_gain(bi_array_excl=left_bi_array)  # IMPLEMENT LATER
    #                 right_local_score = self.MDL_FOIL_gain(bi_array_excl=right_bi_array)
    #
    #                 if left_local_score > 0:
    #                     nml_foil_gain.append(left_local_score)
    #                     info_icol.append(icol)
    #                     info_cut_index.append(i)
    #                     info_cut_type.append(LEFT_CUT)
    #                     info_boolarray.append(left_bi_array)
    #                 if right_local_score > 0:
    #                     nml_foil_gain.append(right_local_score)
    #                     info_icol.append(icol)
    #                     info_cut_index.append(i)
    #                     info_cut_type.append(RIGHT_CUT)
    #                     info_boolarray.append(right_bi_array)
    #
    #                 # if icol == 1 and (cut == 113.5 or cut == 118.5 or cut == 120.5 or cut == 111.5):
    #                 #     print(icol, cut, left_local_score, right_local_score)
    #
    #                 # CHECK whether the beam should be updated
    #                 # if left_local_score > right_local_score:
    #                 #     beam.update(rule_base=self, local_gain=left_local_score, bi_array_excl=left_bi_array,
    #                 #                 icol=icol, var_type=NUMERIC, cut_type=LEFT_CUT, cut=cut,
    #                 #                 excl_or_not=True, bi_array_incl=None, buffer=None)
    #                 # else:
    #                 #     beam.update(rule_base=self, local_gain=right_local_score, bi_array_excl=right_bi_array,
    #                 #                 icol=icol, var_type=NUMERIC, cut_type=RIGHT_CUT, cut=cut,
    #                 #                 excl_or_not=True, bi_array_incl=None, buffer=None)
    #         else:
    #             for i, level in enumerate(self.categorical_levels[icol]):
    #                 within_bi_array = np.isin(self.features_excl_overlap[:, icol], level)
    #                 within_local_score = self.MDL_FOIL_gain(bi_array_excl=within_bi_array)
    #                 # beam.update(rule_base=self, local_gain=within_local_score, bi_array_excl=within_bi_array,
    #                 #             icol=icol, var_type=CATEGORICAL, cut_type=WITHIN_CUT, cut=level,
    #                 #             excl_or_not=True, bi_array_incl=None, buffer=None)
    #                 if within_local_score > 0:
    #                     nml_foil_gain.append(within_local_score)
    #                     info_icol.append(icol)
    #                     info_cut_index.append(i)
    #                     info_cut_type.append(WITHIN_CUT)
    #                     info_boolarray.append(within_bi_array)
    #
    #     nml_foil_gain_sort_index = np.argsort(-np.array(nml_foil_gain))
    #     best_m_nmlfoilgain_index = []
    #     for kk, ind in enumerate(nml_foil_gain_sort_index):
    #         if len(nml_foil_gain_sort_index) >= beam_width:
    #             break
    #
    #         if kk == 0:
    #             best_m_nmlfoilgain_index.append(ind)
    #
    #         for ll, best_ind in enumerate(best_m_nmlfoilgain_index):
    #             jarcard_dist = np.count_nonzero(np.bitwise_or(info_boolarray[best_ind], info_boolarray[ind])) / \
    #                            np.count_nonzero(np.bitwise_and(info_boolarray[best_ind], info_boolarray[ind]))
    #             if jarcard_dist > 0.95:
    #                 break
    #         else:
    #             best_m_nmlfoilgain_index.append(ind)
    #
    #     best_rules = []
    #     for ind in best_m_nmlfoilgain_index:
    #         icol = info_icol[ind]
    #         cut_index = info_cut_index[ind]
    #         cut_type = info_cut_type[ind]
    #
    #         if cut_type == LEFT_CUT:
    #             candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    #                                       (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
    #             candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #             cut = candidate_cuts_icol[cut_index]
    #
    #             bi_array_excl = (self.features_excl_overlap[:, icol] < cut)
    #             bi_array_incl = (self.features[:, icol] < cut)
    #
    #             icols, var_types, cuts, cut_options = list(self.condition.values())
    #             icols = icols + [icol]
    #             var_types = var_types + [NUMERIC]
    #             cuts = cuts + [cut]
    #             cut_options = cut_options + [cut_type]
    #             condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
    #                          "cut_options": cut_options}
    #         elif cut_type == RIGHT_CUT:
    #             candidate_cuts_selector = (candidate_cuts[icol] < np.max(self.features_excl_overlap[:, icol])) & \
    #                                       (candidate_cuts[icol] > np.min(self.features_excl_overlap[:, icol]))
    #             candidate_cuts_icol = candidate_cuts[icol][candidate_cuts_selector]
    #             cut = candidate_cuts_icol[cut_index]
    #
    #             bi_array_excl = (self.features_excl_overlap[:, icol] >= cut)
    #             bi_array_incl = (self.features[:, icol] >= cut)
    #
    #             icols, var_types, cuts, cut_options = list(self.condition.values())
    #             icols = icols + [icol]
    #             var_types = var_types + [NUMERIC]
    #             cuts = cuts + [cut]
    #             cut_options = cut_options + [cut_type]
    #             condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
    #                          "cut_options": cut_options}
    #         else:
    #             level = self.categorical_levels[icol][cut_index]
    #
    #             bi_array_excl = np.isin(self.features_excl_overlap[:, icol], level)
    #             bi_array_incl = np.isin(self.features[:, icol], level)
    #
    #             icols, var_types, cuts, cut_options = list(self.condition.values())
    #             icols = icols + [icol]
    #             var_types = var_types + [CATEGORICAL]
    #             cuts = cuts + [level]
    #             cut_options = cut_options + [cut_type]
    #             condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
    #                          "cut_options": cut_options}
    #
    #         rule = Rule(indices=self.indices[bi_array_incl],
    #                     indices_excl_overlap=self.indices_excl_overlap[bi_array_excl],
    #                     rule_base=self,
    #                     features=self.features[bi_array_incl],
    #                     target=self.target[bi_array_incl],
    #                     features_excl_overlap=self.features_excl_overlap[bi_array_excl],
    #                     target_excl_overlap=self.target_excl_overlap[bi_array_excl],
    #                     data_info=self.data_info,
    #                     condition=condition,
    #                     local_gain=nml_foil_gain[ind])
    #
    #         best_rules.append(rule)
    #
    #     return best_rules

    def MDL_FOIL_gain(self, bi_array_excl, bi_array_incl=None, excl=True):
        """
        :param bi_array_excl: numpy binary array representing a (candidate) refinement of self;
        :return: MDL_FOIL_gain = (NML(target) / len(target) -
        NML(target[bi_arrary]) / len(target[bi_arrya])) / len(target[bi_array])
        """
        candidate_coverage = np.count_nonzero(bi_array_excl)
        regret_refinement = self._regret(N=candidate_coverage, K=self.data_info.num_class)
        if excl:
            p = self._calc_prob_excl(bi_array=bi_array_excl)
            p = p[p != 0]
            neglog_likelihood_refiment = -candidate_coverage * np.sum(p * np.log2(p))
            nml_foil_gain = (self.neglog_likelihood_excl + self.regret_excl) / self.nrow_excl * candidate_coverage - \
                (neglog_likelihood_refiment + regret_refinement)

        else:
            p_incl = self._calc_prob(bi_array=bi_array_incl)
            p_selector = (p_incl != 0)
            p_incl = p_incl[p_selector]

            p_excl = self._calc_prob_excl(bi_array=bi_array_excl)
            p_excl = p_excl[p_selector]  # note that here should NOT be "p_excl != 0"

            neglog_likelihood_refiment = -candidate_coverage * np.sum(p_excl * np.log2(p_incl))
            nml_foil_gain = (self.neglog_likelihood_incl + self.regret_excl) / self.nrow_excl * candidate_coverage - \
                (neglog_likelihood_refiment + regret_refinement)
        return nml_foil_gain

    def _calc_prob(self, bi_array=None, remove_zero=False):
        if bi_array is None:
            p = calc_probs(self.target, self.data_info.num_class)
        else:
            p = calc_probs(self.target[bi_array], self.data_info.num_class)
        if remove_zero:
            p = p[p != 0]
        return p

    def _calc_prob_excl(self, bi_array=None, remove_zero=False):
        if bi_array is None:
            p = calc_probs(self.target_excl_overlap, self.data_info.num_class)
        else:
            p = calc_probs(self.target_excl_overlap[bi_array], self.data_info.num_class)
        if remove_zero:
            p = p[p != 0]
        return p

    def _regret(self, N, K):
        return regret(N, K)


class ElseRule:
    def __init__(self, bool_array, data_info, target, features, get_surrogate_score=False):
        """

        :param bool_array: representing the instances covered by the else-rule, i.e., not covered by any other rule
        :param data_info: meta-data of dataset
        :param target: target for the whole dataset
        :param features: features for the whole dataset
        """
        self.bool_array = bool_array
        self.data_info = data_info
        self.coverage = np.count_nonzero(self.bool_array)

        self.p = self._calc_prob(target[self.bool_array])
        self.neglog_likelihood = self._neglog_likelihood()  # based on the self.p above

        self.regret = self._regret()
        self.nml_score_else_rule = self.neglog_likelihood + self.regret

        self.score = self.regret + self.neglog_likelihood

        if get_surrogate_score:
            self.surrogate_score = self._surrogate_score_else_rule(target=target, features=features)

    def _neglog_likelihood(self):
        p = self.p
        neglog_likelihood = -np.sum(np.log2(p) * p) * self.coverage
        return neglog_likelihood

    def _regret(self):
        return regret(self.coverage, self.data_info.num_class)

    def _calc_prob(self, target):
        p = calc_probs(target, self.data_info.num_class)
        p = p[p != 0]
        return p

    def _surrogate_score_else_rule(self, target, features):
        local_target = target[self.bool_array]
        local_features = features[self.bool_array]
        surrogate_score = surrogate_tree.get_tree_cl(x_train=local_features,
                                                     y_train=local_target,
                                                     num_class=self.data_info.num_class)
        return surrogate_score
