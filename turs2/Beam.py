import numpy as np

class GrowInfoBeam():
    def __init__(self, width):
        self.infos = []
        self.gains = []

        self.worst_gain = None
        self.whichworst_gain = None
        self.width = width

        self.coverage_list = []

    def update(self, info, gain):
        info_coverage = np.count_nonzero(info["incl_bi_array"])
        skip_flag = False
        if info_coverage in self.coverage_list:
            which_equal = self.coverage_list.index(info_coverage)
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

class DiverseCovBeam:
    def __init__(self, width):
        self.coverage_percentage = np.linspace(0, 1, width + 1)[1:]
        self.infos = {}
        for i, cov in enumerate(self.coverage_percentage):
            self.infos[i] = None

        self.gains = {}
        for i, cov in enumerate(self.coverage_percentage):
            self.gains[i] = None

        self.worst_gain = None
        self.width = width


    def update(self, info, gain, coverage_percentage):
        which_coverage_interval = np.searchsorted(a=self.coverage_percentage, v=coverage_percentage)
        if self.infos[which_coverage_interval] is None:
            self.infos[which_coverage_interval] = info
            self.gains[which_coverage_interval] = gain
            self.worst_gain = np.min(self.gains)
        else:
            skip_flag = False
            info_coverage = np.count_nonzero(info["incl_bi_array"])
            if info_coverage == np.count_nonzero(self.infos[which_coverage_interval]["incl_bi_array"]):
                if np.array_equal(info["incl_bi_array"], self.infos[which_coverage_interval]["incl_bi_array"]):
                    skip_flag = True

            if not skip_flag and gain > self.gains[which_coverage_interval]:
                self.infos[which_coverage_interval] = info
                self.gains[which_coverage_interval] = gain
                self.worst_gain = np.min(self.gains)
