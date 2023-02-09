from turs2.Rule import *
from turs2.Beam import *

class Ruleset:
    def __init__(self, data_info):
        self.rules = []

        self.default_p = calc_probs(target=data_info.target, num_class=data_info.num_class)
        self.negloglike = -np.sum(data_info.nrow * np.log2(self.default_p[self.default_p != 0]) * self.default_p[self.default_p != 0])
        self.regret = regret(data_info.nrow, data_info.num_class)
        self.cl_model = 0

        self.total_cl = self.cl_model + self.negloglike + self.regret

        self.data_info = data_info
        self.modelling_groups = [ModellingGroup()]

        self.allrules_neglolglike = 0
        self.allrules_regret = 0

        self.cl_model = 0
        self.else_rule_negloglike = self.negloglike
        self.else_rule_regret = self.regret

        self.uncovered_indices = np.arange(data_info.nrow)

    def fit(self, max_iter=1000):
        # An empty rule with an empty rule set.
        rule = Rule(indices=np.arange(self.data_info.nrow), indices_excl_overlap=self.uncovered_indices,
                    data_info=self.data_info, rule_base=None,
                    condition_matrix=np.repeat(np.nan, self.data_info.ncol * 2).reshape(2, self.data_info.ncol),
                    ruleset=self)
        excl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0)]
        excl_beam_list[0].update(rule=rule, gain=rule.excl_normalized_gain)

        incl_beam_list = [Beam(width=self.data_info.beam_width, rule_length=0),
                          Beam(width=self.data_info.beam_width, rule_length=1)]
        for iter in range(max_iter):
            pass




