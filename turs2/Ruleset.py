from turs2.utils_calculating_cl import *
from turs2.nml_regret import *


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

    def fit(self):
        pass

