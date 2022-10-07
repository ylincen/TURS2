import sys

import numpy as np
from newRule import *
import copy 
import surrogate_tree


class Beam:
    def __init__(self, beam_width):
        self.beam = []  # an array of Rules
        self.nml_foil_gain = []  # an array of local scores (MDL_FOIL_gain)
        self.min_score = 0
        self.arg_min = 0 # the positition of min in the local scores.

        self.beam_width = beam_width

    def update(self, rule_base, local_gain, bi_array_excl, icol, var_type, cut_type, cut,
               excl_or_not, bi_array_incl=None, buffer=None):
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

        if buffer is not None:
            sys.exit("Buffer is deprecated in beam.update().")

        # WHAT IS RULE BASE? ANSWER: the rule base for the candidate growth we are examining now
        # WHAT IS BUFFER? HMM I think we don't need it (July 29)
        if (len(self.beam) < self.beam_width) or (self.min_score < local_gain):
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
            if len(self.beam) < self.beam_width:
                self.beam.append(rule)
                self.nml_foil_gain.append(local_gain)
                if local_gain < self.min_score:
                    self.min_score = local_gain
                    self.arg_min = len(self.nml_foil_gain) - 1
                else:
                    print("_")  # self.min_score and self.arg_min remain unchanged.
            elif self.min_score < local_gain:  # may have problem for grow_incl;
                self.beam[self.arg_min] = rule
                self.nml_foil_gain[self.arg_min] = local_gain
                self.min_score = np.min(self.nml_foil_gain)
                self.arg_min = np.argmin(self.nml_foil_gain)
            else:
                sys.exit("it should not end here!")
                # pass  # it should never end here
        # else:
        #     pass  # do nothing, i.e., no update for beam.

    def from_beam_collect(self, beam_collect):
        for rule in beam_collect.rules:
            self.beam.append(rule)


class BeamCollect:
    """
    What I did in the old-slow implementation:
    1. GrowEx:
        a) start with an empty rule, with nml_foil_gain = 0;
        b) search for the next $m$ ($m$ is beam witdh) best literals by optimizing the nml_foil_gain_EXCL;
        c) repeat b) using beam search, until no new literal can be found to add to the rule;
        d) return the best $m$ rules that have the best surrogate_score_EXCL, with the constraint controlled by
            the "similarity_alpha", the parameter for Diverse Beam Search;
    2. GrowIn:
        a) start with beam obtained in 1.
        b) search for the next $m$ best literals by optimizing the nml_foil_gain_INCL;
        c) repeat b) until no new literal with positive nml_foil_gain can be found;
        d) return the best rule with the best surrogate_score
    """
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