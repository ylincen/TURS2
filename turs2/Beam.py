import numpy as np


# class Beam:
#     def __init__(self, width):
#         self.rules = []
#         self.gains = []
#
#         self.worst_gain = None
#         self.whichworst_gain = None
#         self.width = width
#
#         self.coverage_list = []
#
#     def update(self, rule, gain):
#         if rule in self.rules:
#             pass
#         elif rule.coverage in self.coverage_list:
#             which_equal = self.coverage_list.index(rule.coverage)
#
#             if self.gains[which_equal] < gain:
#                 self.rules[which_equal] = rule
#                 self.gains[which_equal] = gain
#                 self.worst_gain = np.min(self.gains)
#                 self.whichworst_gain = np.argmin(self.gains)
#         else:
#             if len(self.rules) < self.width:
#                 self.rules.append(rule)
#                 self.gains.append(gain)
#                 self.worst_gain = np.min(self.gains)
#                 self.whichworst_gain = np.argmin(self.gains)
#
#                 self.coverage_list.append(rule.coverage)
#             else:
#                 if gain > self.worst_gain:
#                     self.gains.pop(self.whichworst_gain)
#                     self.rules.pop(self.whichworst_gain)
#                     self.rules.append(rule)
#                     self.gains.append(gain)
#                     self.worst_gain = np.min(self.gains)
#                     self.whichworst_gain = np.argmin(self.gains)
#
#                     self.coverage_list.pop(self.whichworst_gain)
#                     self.coverage_list.append(rule.coverage)

class GrowInfoBeam():
    def __init__(self, width):
        self.infos = []
        self.gains = []

        self.worst_gain = None
        self.whichworst_gain = None
        self.width = width

        self.coverage_list = []

    # def update_check_multiple(self, grow_info_list, gain_list):
    #     if isinstance(grow_info_list, list):
    #         assert isinstance(gain_list, list)
    #         for grow_info, gain in zip(grow_info_list, gain_list):
    #             self.update(info=grow_info, gain=gain)
    #     else:
    #         self.update(grow_info_list, gain_list)

    def update(self, info, gain):
        info_coverage = np.count_nonzero(info["incl_bi_array"])
        skip_flag = False
        if info_coverage in self.coverage_list:
            which_equal = self.coverage_list.index(info_coverage)
            # if self.gains[which_equal] < gain:
            #     self.infos[which_equal] = info
            #     self.gains[which_equal] = gain
            #     self.worst_gain = np.min(self.gains)
            #     self.whichworst_gain = np.argmin(self.gains)
            bi_array_in_list = self.infos[which_equal]["incl_bi_array"]
            bi_array_input = info["incl_bi_array"]
            if np.array_equal(bi_array_in_list, bi_array_input):
                skip_flag = True
        if skip_flag is False:
            if len(self.infos) < self.width:
                self.infos.append(info)
                self.gains.append(gain)
                self.worst_gain = np.min(self.gains)
                self.whichworst_gain = np.argmin(self.gains)

                self.coverage_list.append(info_coverage)
            else:
                if gain > self.worst_gain:
                    self.gains.pop(self.whichworst_gain)
                    self.infos.pop(self.whichworst_gain)
                    self.infos.append(info)
                    self.gains.append(gain)
                    self.worst_gain = np.min(self.gains)
                    self.whichworst_gain = np.argmin(self.gains)

                    self.coverage_list.pop(self.whichworst_gain)
                    self.coverage_list.append(info_coverage)
