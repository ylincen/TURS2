import numpy as np
from newRule import *


class Beam:
    def __init__(self, beam_width):
        self.beam = []  # an array of Rules
        self.nml_foil_gain = []  # an array of local scores (MDL_FOIL_gain)
        self.min_score = 0
        self.arg_min = 0 # the positition of min in the local scores.

        self.beam_width = beam_width

    def update(self, rule_base, local_gain, bi_array_excl, icol, var_type, cut_type, cut,
               excl_or_not, bi_array_incl=None, buffer=20):
        """
        Research question: does it make sense to compare different MDL_FOIL_gain with different rule_base????
        Examine a new candidate rule by its local_gain.
        :param rule_base: rule_base for the rule that we are currently examine
        :param local_gain: local score for the rule, either excluding overlap or including overlap, depending on which
        search stage we are in.
        :param bi_array: numpy boolean array for the LOCAL COVER, based on rule_base
        :param buffer: buffer for diverse beam search
        :return:
        """

        if local_gain <= 0:
            return 0   # a "signal" that local_gain is smaller than 0;

        # WHAT IS RULE BASE? ANSWER: the rule base for the candidate growth we are examining now
        # WHAT IS BUFFER? HMM I think we don't need it (July 29)
        if (len(self.beam) < buffer * self.beam_width) or (self.min_score < local_gain):
            # Requirements for updating. Update the beam.
            icols, var_types, cuts, cut_options = list(rule_base.condition.values())
            icols = icols + [icol]
            var_types = var_types + [var_type]
            cuts = cuts + [cut]
            cut_options = cut_options + [cut_type]
            condition = {"icols": icols, "var_types": var_types, "cuts": cuts,
                         "cut_options": cut_options}

            if excl_or_not:  # the local score for the growth is obtained while excluding all overlaps
                if cut_type == LEFT_CUT:
                    bi_array_incl = rule_base.features[:, icol] < cut
                elif cut_type == RIGHT_CUT:
                    bi_array_incl = rule_base.features[:, icol] > cut
                else:  # WITHIN CUT
                    bi_array_incl = np.isin(rule_base.features[:, icol], cut)
                rule = Rule(indices=rule_base.indices[bi_array_incl],
                            indices_excl_overlap=rule_base.indices_excl_overlap[bi_array_excl],
                            rule_base=rule_base,
                            features=rule_base.features[bi_array_incl],
                            target=rule_base.target[bi_array_incl],
                            features_excl_overlap=rule_base.features_excl_overlap[bi_array_excl],
                            target_excl_overlap=rule_base.target_excl_overlap[bi_array_excl],
                            data_info=rule_base.data_info,
                            condition=condition,
                            local_gain=local_gain)
            else:
                rule = Rule(indices=rule_base.indices[bi_array_incl],
                            indices_excl_overlap=rule_base.indices_excl_overlap[bi_array_excl],
                            rule_base=rule_base,
                            features=rule_base.features[bi_array_incl],
                            target=rule_base.target[bi_array_incl],
                            features_excl_overlap=rule_base.features_excl_overlap[bi_array_excl],
                            target_excl_overlap=rule_base.target_excl_overlap[bi_array_excl],
                            data_info=rule_base.data_info,
                            condition=condition,
                            local_gain=local_gain)

            # What's the code below doing??
            # ANS: Depending on whether the len(self.beam) reaches the maximum length
            if len(self.beam) < buffer * self.beam_width:
                self.beam.append(rule)
                self.nml_foil_gain.append(local_gain)
                if local_gain < self.min_score:
                    self.min_score = local_gain
                    self.arg_min = len(self.nml_foil_gain) - 1
                else:
                    pass  # self.min_score and self.arg_min remain unchanged.
            elif self.min_score < local_gain:  # may have problem for grow_incl;
                self.beam[self.arg_min] = rule
                self.nml_foil_gain[self.arg_min] = local_gain
                self.min_score = np.min(self.nml_foil_gain)
                self.arg_min = np.argmin(self.nml_foil_gain)
            else:
                pass  # it should never end here
        else:
            pass  # do nothing, i.e., no update for beam.

    def diversity_prune(self, similarity_alpha):
        rules_with_diversity_constraint, pruned_nml_foil_gain, boolarray_list = [], [], []
        local_gain_ranks = np.argsort(self.nml_foil_gain)  # sort in decrease order
        for i in local_gain_ranks[::-1]:  # iterate from the max
            if len(rules_with_diversity_constraint) == 0:
                rules_with_diversity_constraint.append(self.beam[i])
                pruned_nml_foil_gain.append(self.nml_foil_gain[i])
                boolarray_list.append(self.beam[i].bool_array)
            else:
                # check diversity
                bool_array_i = self.beam[i].bool_array
                for j, r in enumerate(rules_with_diversity_constraint):
                    cover_overlap_count = np.count_nonzero(np.bitwise_and(boolarray_list[j], bool_array_i))
                    cover_union_count = np.count_nonzero(np.bitwise_or(boolarray_list[j], bool_array_i))
                    if cover_overlap_count / cover_union_count > similarity_alpha:
                        break
                else:
                    rules_with_diversity_constraint.append(self.beam[i])
                    pruned_nml_foil_gain.append(self.nml_foil_gain[i])
                    boolarray_list.append(bool_array_i)

            if len(rules_with_diversity_constraint) >= self.beam_width:
                break

        beam = Beam(self.beam_width)
        beam.beam = rules_with_diversity_constraint
        beam.nml_foil_gain = pruned_nml_foil_gain
        beam.beam_width = self.beam_width

        beam.min_score = np.min(pruned_nml_foil_gain)
        beam.arg_min = np.argmin(pruned_nml_foil_gain)

        return beam

    def from_beam_collect(self, beam_collect):
        for rule in beam_collect.rules:
            self.beam.append(rule)


class BeamCollect:
    def __init__(self, beamwidth):
        self.rules = []
        self.scores = []
        self.worst_score = None
        self.arg_worst = 0

        self.beamwidth = beamwidth

    def update(self, rule, score, smaller_is_better):
        if smaller_is_better:
            if len(self.rules) < self.beamwidth:
                self.rules.append(rule)
                self.scores.append(score)
                self.arg_worst = np.argmax(self.scores)

                self.worst_score = self.scores[self.arg_worst]
            elif score < self.worst_score:
                self.rules[self.arg_worst] = rule
                self.scores[self.arg_worst] = score
                self.arg_worst = np.argmax(self.scores)
                self.worst_score = self.scores[self.arg_worst]
            else:
                pass  # Do nothing, and hence beam_collect is not updated;
        else:
            if len(self.rules) < self.beamwidth:
                self.rules.append(rule)
                self.scores.append(score)
                self.arg_worst = np.argmin(self.scores)

                self.worst_score = self.scores[self.arg_worst]
            elif score > self.worst_score:
                self.rules[self.arg_worst] = rule
                self.scores[self.arg_worst] = score
                self.arg_worst = np.argmin(self.scores)
                self.worst_score = self.scores[self.arg_worst]
            else:
                pass  # Do nothing, and hence beam_collect is not updated;







