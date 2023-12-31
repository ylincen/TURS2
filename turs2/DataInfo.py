import sys
from collections import namedtuple
import platform
import pandas as pd
import numpy as np
from turs2.constant import *
from math import log
from scipy.special import gammaln, comb
from turs2.utils_modelencoding import *
from turs2.utils_calculating_cl import *
from turs2.utils_namedtuple import AlgConfig

from datetime import datetime
import os


class DataInfo:
    def __init__(self, X, y, beam_width=None, alg_config=None, not_use_excl_=False):
        if alg_config is None:
            assert beam_width is not None
            self.alg_config = AlgConfig(
                num_candidate_cuts=100, max_num_rules=500, max_grow_iter=200, num_class_as_given=None,
                beam_width=beam_width,
                log_learning_process=False,
                dataset_name=None,
                feature_names=["X" + str(i) for i in range(X.shape[1])],
                validity_check="either"
            )
        else:
            self.alg_config = alg_config

        self.not_use_excl_ = not_use_excl_

        if type(X) != np.ndarray:
            self.features = X.to_numpy()
        else:
            self.features = X

        if type(y) != np.ndarray:
            self.target = y.to_numpy().flatten()
        else:
            self.target = y

        self.max_grow_iter = self.alg_config.max_grow_iter
        self.nrow, self.ncol = X.shape[0], X.shape[1]

        self.cached_number_of_rules_for_cl_model = self.alg_config.max_grow_iter  # for cl_model

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(self.target))
        if self.alg_config.num_class_as_given is not None:
            self.num_class = self.alg_config.num_class_as_given

        self.num_candidate_cuts = self.alg_config.num_candidate_cuts

        # get_candidate_cuts (for NUMERIC only, CATEGORICAL features need to be one-hot-encoded)
        self.candidate_cuts = self.candidate_cuts_quantile_mid_points(self.num_candidate_cuts)

        self.feature_names = self.alg_config.feature_names
        self.beam_width = self.alg_config.beam_width
        self.dataset_name = self.alg_config.dataset_name
        self.default_p = calc_probs(self.target, self.num_class)

        self.log_learning_process = self.alg_config.log_learning_process

        if self.log_learning_process is True:
            assert self.alg_config.X_test is not None
            assert self.alg_config.y_test is not None
            self.X_test, self.y_test = self.alg_config.X_test, self.alg_config.y_test


            if self.alg_config.rf_assist is True:
                assert self.alg_config.rf_oob_decision_function is not None
                self.rf_oob_decision_ = self.alg_config.rf_oob_decision_function
                log_folder_name = "log_results"
                if self.alg_config.dataset_name is not None:
                    log_folder_name = log_folder_name + "_" + self.alg_config.dataset_name
                else:
                    log_folder_name = log_folder_name + "_" + "UnnamedData_"

                date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_folder_name = log_folder_name + "_" + date_and_time
                try:
                    os.makedirs(log_folder_name)
                    print("Directory", log_folder_name, "Created.")
                except FileExistsError:
                    print("Directory", log_folder_name, "already exists.")

                log_datainfo = "Parameters for this class: num_candidate_cuts, max_rule_length, feature_names, beam_width"
                log_datainfo += "\n " + str([self.alg_config.num_candidate_cuts, self.alg_config.max_rule_length,
                                             self.alg_config.feature_names, self.alg_config.beam_width])

                log_datainfo += "\n candidate cuts for each dimension: \n"
                for j, v in self.candidate_cuts.items():
                    log_datainfo += str(self.candidate_cuts[j]) + "\n"

                system_name = platform.system()
                if system_name == "Windows":
                    with open(log_folder_name + "\\datainfo.txt", "w") as flog:
                        flog.write(log_datainfo)
                    self.log_folder_name = log_folder_name
                else:
                    with open(log_folder_name + "/datainfo.txt", "w") as flog:
                        flog.write(log_datainfo)
                    self.log_folder_name = log_folder_name

    def get_candidate_cuts_CLASSY(self, num_candidate_cuts):
        candidate_cuts = {}

        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                quantile_percentage = [1 / (num_candidate_cuts_i + 1) * ncut for ncut in range(0, num_candidate_cuts_i + 2)]
                value_quantiles = np.nanquantile(feature, quantile_percentage, interpolation='midpoint')[1:-1]
                value_quantiles = np.unique(value_quantiles)
                candidate_cuts[i] = value_quantiles
        return candidate_cuts

    def get_candidate_cuts_quantile(self, num_candidate_cuts):
        candidate_cuts = {}

        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                quantile_percentage = np.linspace(0, 1, num_candidate_cuts_i + 2)[1:-1]
                candidate_cuts[i] = np.quantile(feature, quantile_percentage)
        return candidate_cuts

    def get_candidate_cuts_indep_data(self, num_candidate_cuts):
        candidate_cuts = {}
        for i, feature in enumerate(self.features.T):
            unique_value = np.unique(feature)

            num_candidate_cuts_i = min(num_candidate_cuts, len(unique_value) - 1)

            if len(unique_value) < 2:
                candidate_cuts[i] = np.array([], dtype=float)
            else:
                candidate_cuts[i] = np.linspace(unique_value[0], unique_value[-1], num_candidate_cuts_i)
        return candidate_cuts

    def get_candidate_cuts(self, num_candidate_cuts):
        candidate_cuts = {}
        dim_iter_counter = -1

        if num_candidate_cuts is list:
            pass
        else:
            num_candidate_cuts = np.repeat(num_candidate_cuts, self.ncol)

        for i, feature in enumerate(self.features.T):
            dim_iter_counter += 1

            sort_feature = np.unique(feature)
            if len(sort_feature) <= 1:
                candidate_cut_this_dimension = np.array([], dtype=float)
                candidate_cuts[i] = candidate_cut_this_dimension
            else:
                candidate_cut_this_dimension = \
                    (sort_feature[0:(len(sort_feature) - 1)] + sort_feature[1:len(sort_feature)]) / 2
                # to set the bins for each numeric dimension
                if (num_candidate_cuts[i] > 1) & (len(candidate_cut_this_dimension) > num_candidate_cuts[i]):
                    select_indices = np.linspace(0, len(candidate_cut_this_dimension) - 1, num_candidate_cuts[i] + 1,
                                                 endpoint=True, dtype=int)
                    select_indices = select_indices[
                                     1:(len(select_indices) - 1)]  # remove the start and end point
                    candidate_cuts[i] = candidate_cut_this_dimension[select_indices]
                else:
                    candidate_cuts[i] = candidate_cut_this_dimension

        return candidate_cuts

    def candidate_cuts_quantile_mid_points(self, num_candidate_cuts):
        candidate_cuts = {}
        dim_iter_counter = -1

        if num_candidate_cuts is list:
            pass
        else:
            num_candidate_cuts = np.repeat(num_candidate_cuts, self.ncol)

        for i, feature in enumerate(self.features.T):
            dim_iter_counter += 1

            sort_feature = np.sort(feature + np.random.random(len(feature)) * 0.000001)
            unique_feature = np.unique(feature)
            if len(unique_feature) <= 1:
                candidate_cut_this_dimension = np.array([], dtype=float)
                candidate_cuts[i] = candidate_cut_this_dimension
            elif np.array_equal(unique_feature, np.array([0.0, 1.0])):
                candidate_cuts[i] = np.array([0.5])
            else:
                candidate_cut_this_dimension = \
                    (sort_feature[0:(len(sort_feature) - 1)] + sort_feature[1:len(sort_feature)]) / 2

                num_candidate_cuts_i = np.min([len(unique_feature) - 1, num_candidate_cuts[i]])

                if (num_candidate_cuts[i] > 1) & (len(candidate_cut_this_dimension) > num_candidate_cuts_i):
                    select_indices = np.linspace(0, len(candidate_cut_this_dimension) - 1, num_candidate_cuts_i + 2,
                                                 endpoint=True, dtype=int)
                    select_indices = select_indices[
                                     1:(len(select_indices) - 1)]  # remove the start and end point
                    candidate_cuts[i] = candidate_cut_this_dimension[select_indices]
                else:
                    candidate_cuts[i] = candidate_cut_this_dimension

        return candidate_cuts
