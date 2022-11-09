from utils import *
from nml_regret import *
import surrogate_tree


class ElseRule:
    def __init__(self, bool_array, data_info, target, features, get_surrogate_score=False):
        """

        :param bool_array: representing the instances covered by the else-rule, i.e., not covered by any other rule
        :param data_info: meta-data of dataset
        :param target: target for the whole dataset
        :param features: features for the whole dataset
        """
        self.bool_array = bool_array  # for the cover
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
        p = p[p != 0]

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
