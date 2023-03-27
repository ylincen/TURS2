import numpy as np


class Beam:
    def __init__(self, width, rule_length):
        self.rules = []
        self.gains = []

        self.worst_gain = None
        self.whichworst_gain = None
        self.width = width

        # rules with the same length should be put into the same Beam (ignoring redundant literals, i.e., X_1 > 1 and X_1 > 2 is a rule with length 2)
        self.rule_length = rule_length

    def update(self, rule, gain):
        if len(self.rules) < self.width:
            if gain > 0 or len(rule.icols_in_order) == 0:  # we always add the empty rule to start
                self.rules.append(rule)
                self.gains.append(gain)
                self.worst_gain = np.min(self.gains)
                self.whichworst_gain = np.argmin(self.gains)
        else:
            if gain > self.worst_gain:
                self.gains.pop(self.whichworst_gain)
                self.rules.pop(self.whichworst_gain)
                self.rules.append(rule)
                self.gains.append(gain)
                self.worst_gain = np.min(self.gains)
                self.whichworst_gain = np.argmin(self.gains)


