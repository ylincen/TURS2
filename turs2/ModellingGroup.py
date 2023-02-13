import numpy as np
from turs2.utils_calculating_cl import *


class ModellingGroup:
    def __init__(self, data_info, bool_cover, bool_use_for_model, rules_involved, prob_model, prob_cover):
        self.data_info = data_info
        self.bool_cover = bool_cover  # intersection of rules
        self.bool_model = bool_use_for_model  # union of rules
        self.rules_involvde = rules_involved

        self.cover_count = np.count_nonzero(bool_cover)
        self.use_for_model_count = np.count_nonzero(bool_use_for_model)

        self.prob_model = prob_model
        self.prob_cover = prob_cover
        self.negloglike = -self.cover_count * np.sum(prob_cover[prob_model != 0] * np.log2(prob_model[prob_model != 0]))

    def evaluate_rule_approximate(self, rule):
        new_bool_cover = np.bitwise_and(rule.bool_array, self.bool_cover)
        nonRule_cover = np.bitwise_and(~rule.bool_array, self.bool_cover)

        new_cover_count = np.count_nonzero(new_bool_cover)
        if new_cover_count == 0:
            return self.negloglike
        else:
            new_prob_cover = calc_probs(self.data_info.target[new_bool_cover], self.data_info.num_class)

            # The probability of the intersection part is approximately evaluated as the weighted average
            weighted_p = self.use_for_model_count * self.prob_model + rule.prob * rule.coverage
            negloglike_rule_and_mg = (
                    -new_cover_count *
                    np.sum(
                        np.log2(weighted_p[weighted_p != 0]) * new_prob_cover[weighted_p != 0]
                    )
            )

            nonRule_cover_count = np.count_nonzero(nonRule_cover)
            if nonRule_cover_count > 0:
                prob_nonRule = calc_probs(self.data_info.target[nonRule_cover], self.data_info.num_class)
                negloglike_NonRule_and_mg = (
                    -nonRule_cover_count *
                    np.sum(
                        np.log2(self.prob_model[self.prob_model != 0]) * prob_nonRule[self.prob_model != 0]
                    )
                )
            else:
                negloglike_NonRule_and_mg = 0

            return negloglike_rule_and_mg + negloglike_NonRule_and_mg



