from utils import *
from nml_regret import *
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

        p_selector2 = (self.prob != 0)
        p_selector = (self.prob_excl != 0)

        # if self.rule_base is None:
        #     self.neglog_likelihood_incl = -np.sum(self.prob_excl[p_selector2] *
        #                                       np.log2(self.prob[p_selector2])) * self.nrow_excl
        #     self.neglog_likelihood_excl = -np.sum(self.prob_excl[p_selector2] *
        #                                       np.log2(self.prob[p_selector2])) * self.nrow_excl
        # else:
        #     # code length of encoding nrow_excl instances by prob_excl
        #     self.neglog_likelihood_excl = \
        #         -np.sum(self.prob_excl[p_selector] *
        #                 np.log2(self.prob_excl[p_selector])) * self.nrow_excl
        #
        #     # code length of encoding nrow_excl instances by prob
        #     self.neglog_likelihood_incl = -np.sum(self.prob_excl[p_selector2] *
        #                                           np.log2(self.prob[p_selector2])) * self.nrow_excl

        self.neglog_likelihood_excl = \
            -np.sum(self.prob_excl[p_selector] *
                    np.log2(self.prob_excl[p_selector])) * self.nrow_excl

        # code length of encoding nrow_excl instances by prob
        self.neglog_likelihood_incl = -np.sum(self.prob_excl[p_selector2] *
                                              np.log2(self.prob[p_selector2])) * self.nrow_excl

        if len(indices) == len(indices_excl_overlap):
            search_phase = 1
        else:
            search_phase = 2

        if self.rule_base is None:
            self.cl_model = self.get_cl_model(search_phase=search_phase)
        else:
            self.cl_model = self.get_cl_model(search_phase=search_phase) + self.rule_base.cl_model

        self.local_gain = local_gain

    def get_cl_model(self, search_phase):
        if search_phase == 2:
            if self.rule_base is None:
                features = self.features
            else:
                features = self.rule_base.features
        else:
            if self.rule_base is None:
                features = self.features_excl_overlap
            else:
                features = self.rule_base.features_excl_overlap
        icols = self.condition["icols"]
        cut_options = self.condition["cut_options"]

        if len(icols) == 0:
            cl_model = 0
        else:
            icol = self.condition["icols"][-1]
            cut_option = self.condition["cut_options"][-1]

            if cut_option == WITHIN_CUT:
                if icol in icols[:len(icols)-1]:
                    cl_model = 0
                else:
                    # print(len(self.categorical_levels[icol]))
                    if len(self.categorical_levels[icol]) == 0:
                        cl_model = 0
                    else:
                        cl_model = np.log2(len(self.categorical_levels[icol]))
            else:
                if icol in icols[:len(icols)-1]:

                    update_cl_model = True
                    for kcol, cut_option_k in zip(icols, cut_options):
                        if kcol == icol and cut_option_k == cut_option:
                            cl_model = 0
                            update_cl_model = False
                            break

                    if update_cl_model:
                        shrinking_factor = len(features[:, icol]) / len(self.data_info.target)
                        cl_model = np.log2(len(self.data_info.candidate_cuts[icol]) * shrinking_factor)
                else:
                    shrinking_factor = len(features[:, icol]) / len(self.data_info.target)
                    cl_model = np.log2(len(self.data_info.candidate_cuts[icol]) * shrinking_factor)

        return cl_model

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
                # if len(categorical_levels[icol]) == 0:
                #     print("here")

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

    def mdl_gain(self, bi_array_excl, bi_array_incl=None, excl=True):
        if excl:
            candidate_coverage = np.count_nonzero(bi_array_excl)
            p = self._calc_prob_excl(bi_array=bi_array_excl)
            neglog_likelihood_refiment = -candidate_coverage * np.sum(p[p != 0] * np.log2(p[p != 0]))
            neglog_likelihood_previous = -candidate_coverage * np.sum(np.log2(self.prob_excl[p != 0]) * p[p != 0])

            regret_refiment = self._regret(N=candidate_coverage, K=self.data_info.num_class)
            regret_previous = self._regret(N=self.nrow_excl, K=self.data_info.num_class) / self.nrow_excl * candidate_coverage
        else:
            candidate_coverage = np.count_nonzero(bi_array_incl)
            p = self._calc_prob(bi_array=bi_array_incl)
            neglog_likelihood_refiment = -candidate_coverage * np.sum(p[p != 0] * np.log2(p[p != 0]))
            neglog_likelihood_previous = -candidate_coverage * np.sum(np.log2(self.prob[p != 0]) * p[p != 0])  # BE CAREFUL HERE FOR THE ELSE_RULE!!!

            regret_refiment = self._regret(N=candidate_coverage, K=self.data_info.num_class)
            regret_previous = self._regret(N=self.nrow_excl, K=self.data_info.num_class) / self.nrow_excl * candidate_coverage
        mdl_gain = neglog_likelihood_previous + regret_previous - (neglog_likelihood_refiment + regret_refiment)

        return mdl_gain

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
