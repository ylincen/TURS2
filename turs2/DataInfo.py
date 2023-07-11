import sys
from collections import namedtuple
import platform
import pandas as pd
import numpy as np
from constant import *
from math import log
from scipy.special import gammaln, comb
from turs2.utils_modelencoding import *
from turs2.utils_calculating_cl import *
from turs2.utils_alg_config import AlgConfig

from datetime import datetime
import os


class DataInfo:
    def __init__(self, X, y, beam_width, alg_config=None, log_folder_name="log_withoutDateTime"):
        if alg_config is None:
            alg_config = AlgConfig(
                num_candidate_cuts=100, max_num_rules=500, max_grow_iter=200, num_class_as_given=None,
                beam_width=beam_width,
                log_learning_process=False, log_folder_name=log_folder_name,
                dataset_name=None, X_test=None, y_test=None, # X_test, y_test only for logging
                rf_assist=False, rf_oob_decision_function=None,
                feature_names=["X" + str(i) for i in range(X.shape[1])],
                beamsearch_positive_gain_only=False, beamsearch_normalized_gain_must_increase_comparing_rulebase=False,
                beamsearch_stopping_when_best_normalized_gain_decrease=False,
                validity_check="incl_check", rerun_on_invalid=False, rerun_positive_control=False
            )
            self.alg_config = alg_config
        else:
            self.alg_config = alg_config

        if type(X) != np.ndarray:
            self.features = X.to_numpy()
        else:
            self.features = X

        if type(y) != np.ndarray:
            self.target = y.to_numpy().flatten()
        else:
            self.target = y
        if self.alg_config.log_learning_process:
            self.log_folder_name = self.alg_config.log_folder_name
            os.makedirs(self.alg_config.log_folder_name, exist_ok=True)
        self.max_rule_length = self.alg_config.max_grow_iter  # TODO: the name max_rule_length is misleading
        self.nrow, self.ncol = X.shape[0], X.shape[1]

        self.cached_number_of_rules_for_cl_model = self.alg_config.max_num_rules  # for cl_model

        # get num_class, ncol, nrow,
        self.num_class = len(np.unique(self.target))
        if self.alg_config.num_class_as_given is not None:
            self.num_class = self.alg_config.num_class_as_given

        self.num_candidate_cuts = self.alg_config.num_candidate_cuts
        # get_candidate_cuts (for NUMERIC only, CATEGORICAL features need to be one-hot-encoded)
        # self.candidate_cuts = self.get_candidate_cuts_indep_data(num_candidate_cuts)
        self.candidate_cuts = self.candidate_cuts_quantile_mid_points(self.num_candidate_cuts)

        self.feature_names = self.alg_config.feature_names
        self.beam_width = self.alg_config.beam_width
        self.dataset_name = self.alg_config.dataset_name
        self.default_p = calc_probs(self.target, self.num_class)

        self.log_learning_process = self.alg_config.log_learning_process

        self.rf_oob_decision_ = self.alg_config.rf_oob_decision_function

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

    # def __init__(self, X, y, num_candidate_cuts, max_rule_length, feature_names, beam_width, dataset_name=None,
    #              X_test=None, y_test=None, rf_oob_decision_=None, log_learning_process=True, num_class_given=None):
    #
    #     """
    #     Meta-data for an input data
    #     data: pandas data frame
    #     features: feature matrix in numpy nd array
    #     target: target variable in numpy 1d array
    #     """
    #     self.X_test, self.y_test = X_test, y_test
    #     self.rf_oob_decision_ = rf_oob_decision_   # used for random-forest-assisted rule learning
    #
    #     if type(X) != np.ndarray:
    #         self.features = X.to_numpy()
    #     else:
    #         self.features = X
    #
    #     if type(y) != np.ndarray:
    #         self.target = y.to_numpy().flatten()
    #     else:
    #         self.target = y
    #
    #     self.max_rule_length = max_rule_length
    #     self.nrow, self.ncol = X.shape[0], X.shape[1]
    #     self.cached_number_of_rules_for_cl_model = 100  # for cl_model
    #
    #     # get num_class, ncol, nrow,
    #     self.num_class = len(np.unique(self.target))
    #     if num_class_given is not None:
    #         self.num_class = num_class_given
    #
    #     self.num_candidate_cuts = num_candidate_cuts
    #     # get_candidate_cuts (for NUMERIC only; CATEGORICAL dims will do rule.get_categorical_values)
    #     # self.candidate_cuts = self.get_candidate_cuts_CLASSY(num_candidate_cuts)
    #     # self.candidate_cuts = self.get_candidate_cuts(num_candidate_cuts)
    #     # self.candidate_cuts = self.get_candidate_cuts_indep_data(num_candidate_cuts)
    #     # self.candidate_cuts = self.get_candidate_cuts_quantile(num_candidate_cuts)
    #     self.candidate_cuts = self.candidate_cuts_quantile_mid_points(num_candidate_cuts)
    #
    #     self.feature_names = feature_names
    #     self.beam_width = beam_width
    #
    #     self.dataset_name = dataset_name
    #
    #     self.default_p = calc_probs(self.target, self.num_class)
    #
    #     self.log_learning_process = log_learning_process
    #     if log_learning_process:
    #         log_folder_name = "log_results"
    #         if dataset_name is not None:
    #             log_folder_name = log_folder_name + "_" + dataset_name
    #         date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         log_folder_name = log_folder_name + "_" + date_and_time
    #         try:
    #             os.makedirs(log_folder_name)
    #             print("Directory", log_folder_name, "Created.")
    #         except FileExistsError:
    #             print("Directory", log_folder_name, "already exists.")
    #
    #         log_datainfo = "Parameters for this class: num_candidate_cuts, max_rule_length, feature_names, beam_width"
    #         log_datainfo += "\n " + str([num_candidate_cuts, max_rule_length, feature_names, beam_width])
    #
    #         log_datainfo += "\n candidate cuts for each dimension: \n"
    #         for j, v in self.candidate_cuts.items():
    #             log_datainfo += str(self.candidate_cuts[j]) + "\n"
    #
    #         system_name = platform.system()
    #         if system_name == "Windows":
    #             with open(log_folder_name + "\\datainfo.txt", "w") as flog:
    #                 flog.write(log_datainfo)
    #             self.log_folder_name = log_folder_name
    #         else:
    #             with open(log_folder_name + "/datainfo.txt", "w") as flog:
    #                 flog.write(log_datainfo)
    #             self.log_folder_name = log_folder_name

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
