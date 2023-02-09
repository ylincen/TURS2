import numpy as np


class Beam:
    def __init__(self, width, rule_length):
        self.rules = []
        self.gains = []

        # self.best_gain = None
        # self.whichbest_gain = None

        self.worst_gain = None
        self.whichworst_gain = None
        self.width = width

        # rules with the same length should be put into the same Beam (ignoring redundant literals, i.e., X_1 > 1 and X_1 > 2 is a rule with length 2)
        self.rule_length = rule_length

    def update(self, rule, gain):
        if len(self.rules) >= self.width:
            self.gains.pop(self.whichworst_gain)
            self.rules.pop(self.whichworst_gain)

        self.rules.append(rule)
        self.gains.append(gain)
        self.worst_gain = np.min(gain)
        self.whichworst_gain = np.argmin(gain)




