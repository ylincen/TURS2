import sys
import pandas as pd
import numpy as np
from constant import *


class DataInfo:
    def __init__(self, data=None, features=None, target=None, max_cat_level=None, max_bin_num=None):
        """
        Meta-data for an input data
        data: pandas data frame
        features: feature matrix in numpy nd array
        target: target variable in numpy 1d array
        """
        if (features is None or target is None) and (data is None):
            sys.exit("Error: data / features & target are all None!")

        self.data = data
        self.nrow, self.ncol = data.shape
        self.ncol = self.ncol - 1  # do not count the target column

        if data is not None:
            features, target = self.from_pd_data()

        self.features = features
        self.target = target

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(target))

        # Default parameters for deciding the variable types: categorical and numeric
        if max_cat_level is None:
            max_cat_level = 5
        # Default parameters for deciding the number of cut points for NUMERIC variables
        if max_bin_num is None:
            max_bin_num = 100

        # get dim_type, i.e., CATEGORICAL OR NUMERIC for each dimension (column) of featuress
        self.dim_type = self.get_dim_type(max_cat_level)

        # get_candidate_cuts (for NUMERIC only; CATEGORICAL dims will do rule.get_categorical_values)
        self.candidate_cuts_for_cl, self.candidate_cuts = self.get_candidate_cuts(max_bin_num)

    def from_pd_data(self):
        features = self.data.iloc[:, :self.ncol].to_numpy()
        target = self.data.iloc[:, self.ncol].to_numpy()
        return [features, target]

    def get_dim_type(self, max_cat_level):
        var_types = np.zeros(self.ncol, dtype=int)
        for i in range(self.ncol):
            if len(np.unique(self.features[:, i])) < max_cat_level:
                var_types[i] = CATEGORICAL
            else:
                var_types[i] = NUMERIC
        return var_types

    def get_candidate_cuts(self, max_bin_num):
        candidate_cuts = {}  # for calculating the code length of model
        candidate_cuts_search = {}
        dim_iter_counter = -1

        if max_bin_num is list:
            pass
        else:
            max_bin_num = np.repeat(max_bin_num, self.ncol)

        for i, feature in enumerate(self.features.T):
            dim_iter_counter += 1
            if self.dim_type[i] == CATEGORICAL:
                pass
            else:
                sort_feature = np.unique(feature)
                if len(sort_feature) <= 1:
                    candidate_cut_this_dimension = np.array([], dtype=float)
                    candidate_cuts[i] = candidate_cut_this_dimension
                    candidate_cuts_search[i] = candidate_cut_this_dimension
                else:
                    candidate_cut_this_dimension = (sort_feature[0:(len(sort_feature) - 1)] +
                                                    sort_feature[1:len(sort_feature)]) / 2
                    # to set the bins for each numeric dimension
                    if (max_bin_num[i] > 1) & (len(candidate_cut_this_dimension) > max_bin_num[i]):
                        select_indices = np.linspace(0, len(candidate_cut_this_dimension) - 1, max_bin_num[i] + 1,
                                                     endpoint=True, dtype=int)
                        select_indices = select_indices[
                                         1:(len(select_indices) - 1)]  # remove the start and end point
                        candidate_cuts[i] = candidate_cut_this_dimension
                        candidate_cuts_search[i] = candidate_cut_this_dimension[select_indices]
                    else:
                        candidate_cuts[i] = candidate_cut_this_dimension
                        candidate_cuts_search[i] = candidate_cut_this_dimension

        return [candidate_cuts, candidate_cuts_search]
