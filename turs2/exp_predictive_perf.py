import sys

# for mac local
sys.path.extend(['/Users/yanglincen/projects/TURS'])
sys.path.extend(['/Users/yanglincen/projects/TURS/turs2'])
# for DSlab server:
sys.path.extend(['/home/yangl3/projects/turs'])
sys.path.extend(['/home/yangl3/projects/turs/turs2'])

import numpy as np
import pandas as pd
import copy
import time
import cProfile
from datetime import datetime
# from line_profiler import LineProfiler
import cProfile, pstats

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

from turs2.exp_utils import *

np.seterr(all='raise')

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def read_data(data_name, datasets_without_header_row, datasets_with_header_row, folder_name="./datasets/"):
    data_path = folder_name + data_name + ".csv"
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
    if data_name == "anuran":
        d = d.iloc[:, 1:]
    return d

def preprocess_data(d, colnames=None, threshold_categorical=20):
    # threshold_categorical=20 is what we used in the JMLR paper
    le = LabelEncoder()
    d.iloc[:, -1] = le.fit_transform(d.iloc[:, -1])

    if colnames is not None:
        new_colnames = []

    le_feature = OneHotEncoder(sparse=False, dtype=int, drop="if_binary")

    for icol in range(d.shape[1] - 1):
        if d.iloc[:, icol].dtype == "float":
            d_transformed = d.iloc[:, icol]
            if colnames is not None:
                new_colnames.append(colnames[icol])
        elif d.iloc[:, icol].dtype == "int" and len(np.unique(d.iloc[:, icol])) > threshold_categorical:
            d_transformed = d.iloc[:, icol]
            if colnames is not None:
                new_colnames.append(colnames[icol])
        else:
            d_transformed = le_feature.fit_transform(d.iloc[:, icol:(icol+1)])
            d_transformed = pd.DataFrame(d_transformed)
            if colnames is not None:
                new_colnames.extend([colnames[icol] + "_" + str(unique_) for unique_ in np.unique(d.iloc[:, icol])])
        if icol == 0:
            d_feature = d_transformed
        else:
            d_feature = pd.concat([d_feature, d_transformed], axis=1)

    d = pd.concat([d_feature, d.iloc[:, -1]], axis=1)
    if colnames is not None:
        d.columns = new_colnames + [colnames[-1]]
    else:
        d.columns = ["X" + str(i) for i in range(d.shape[1])]
    return d

def calculate_roc_auc_and_logloss(ruleset, y_test, y_pred_prob, y_train, y_pred_prob_train):
    if ruleset.data_info.num_class == 2:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train[:, 1])

        logloss_train = log_loss(y_train, y_pred_prob_train[:, 1])
        logloss_test = log_loss(y_test, y_pred_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train, multi_class="ovr")

        logloss_train = log_loss(y_train, y_pred_prob_train)
        logloss_test = log_loss(y_test, y_pred_prob)
    return [roc_auc, roc_auc_train, logloss_test, logloss_train]

def calculate_rule_lengths(ruleset):
    # rule lengths
    rule_lengths = []
    for r in ruleset.rules:
        r_len = np.count_nonzero(r.condition_count)
        rule_lengths.append(r_len)
    return np.mean(rule_lengths)

def calculate_brier_and_prauc(ruleset, y_train, y_test, y_pred_prob, y_pred_prob_train):
    # multi-class macro PR AUC, and (multi-class) Brier score
    Brier_train, Brier_test = 0, 0
    pr_auc_train, pr_auc_test = 0, 0
    y_unique = np.arange(ruleset.data_info.num_class)  # we made sure that y_unique is always in the form of [0,1,2,..]

    if len(y_unique) == 2:
        y_unique = np.array([1], dtype=int)

    for yy in y_unique:
        positive_mask_train = (y_train == yy)
        positive_mask_test = (y_test == yy)

        Brier_train += np.sum((y_pred_prob_train[:, yy] - positive_mask_train) ** 2)
        Brier_test += np.sum((y_pred_prob[:, yy] - positive_mask_test) ** 2)
        Brier_train = Brier_train / len(y_pred_prob_train)
        Brier_test = Brier_test / len(y_pred_prob)

        pr_train = precision_recall_curve(positive_mask_train, y_pred_prob_train[:, yy])
        pr_test = precision_recall_curve(positive_mask_test, y_pred_prob[:, yy])

        pr_auc_train += auc(pr_train[1], pr_train[0])
        pr_auc_test += auc(pr_test[1], pr_test[0])
    return [pr_auc_test/len(y_unique), pr_auc_train/len(y_unique), Brier_test, Brier_train]

def calculate_train_test_prob_diff(ruleset, X_test, y_test, return_for_all_rules=False):
    # weighted average of train_test prob. est. difference for each rule,
    rule_test_prob_info = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rules_test_prob = rule_test_prob_info["rules_test_p"]
    rules_test_prob.append(rule_test_prob_info["else_rule_p"])
    rules_train_prob = [r.prob for r in ruleset.rules]
    rules_train_prob.append(ruleset.else_rule_p)
    rules_coverage_including_else = [r.coverage for r in ruleset.rules]
    rules_coverage_including_else.append(ruleset.else_rule_coverage)
    p_diff_rules = []
    for tr_p_, test_p_ in zip(rules_train_prob, rules_test_prob):
        p_diff = np.mean(abs(tr_p_ - test_p_))
        p_diff_rules.append(p_diff)

    train_test_prob_diff = np.average(p_diff_rules, weights=rules_coverage_including_else)
    if return_for_all_rules:
        return [p_diff_rules, train_test_prob_diff, rules_test_prob, rules_train_prob]
    else:
        return train_test_prob_diff

def calculate_random_picking_pred_performance(ruleset, X, y, num_repetition, cover_mat):
    random_picking_roc_auc = []
    random_picking_pr_auc = []
    random_picking_brier_score = []
    random_picking_logloss = []

    for _ in range(num_repetition):
        y_pred_prob = predict_random_picking_for_overlaps(ruleset=ruleset, X=X, seed=_, cover_mat=cover_mat)
        roc_auc, roc_auc_train, logloss_test, logloss_train = \
            calculate_roc_auc_and_logloss(ruleset=ruleset, y_test=y, y_pred_prob=y_pred_prob,
                                          y_train=y, y_pred_prob_train=y_pred_prob)
        pr_auc_test, pr_auc_train, Brier_test, Brier_train = \
            calculate_brier_and_prauc(ruleset=ruleset, y_train=y, y_test=y, y_pred_prob=y_pred_prob,
                                      y_pred_prob_train=y_pred_prob)
        random_picking_roc_auc.append(roc_auc)
        random_picking_pr_auc.append(pr_auc_test)
        random_picking_brier_score.append(Brier_test)
        random_picking_logloss.append(logloss_test)
    return [np.mean(random_picking_roc_auc), np.mean(random_picking_pr_auc),
            np.mean(random_picking_brier_score), np.mean(random_picking_logloss)]

def calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time):
    res = predict_ruleset(ruleset, X_test, y_test)
    res_train = predict_ruleset(ruleset, X_train, y_train)
    cover_mat_test = cover_matrix_fun(ruleset, X_test)
    cover_mat_train = cover_matrix_fun(ruleset, X_train)

    roc_auc, roc_auc_train, logloss_test, logloss_train = \
        calculate_roc_auc_and_logloss(ruleset=ruleset, y_test=y_test, y_train=y_train,
                                      y_pred_prob=res, y_pred_prob_train=res_train)

    pr_auc_test, pr_auc_train, Brier_test, Brier_train = \
        calculate_brier_and_prauc(ruleset=ruleset, y_test=y_test, y_train=y_train,
                                  y_pred_prob_train=res_train, y_pred_prob=res)

    accuracy_test = np.mean(np.argmax(res, axis=1) == y_test)
    accuracy_train = np.mean(np.argmax(res_train, axis=1) == y_train)

    avg_rule_length = calculate_rule_lengths(ruleset)

    p_diff_rules, train_test_prob_diff, rules_test_prob, rules_train_prob \
        = calculate_train_test_prob_diff(ruleset, X_test, y_test, return_for_all_rules=True)

    avg_num_literals_for_each_datapoint = explainability_analysis(ruleset, X_test, cover_mat_test)  # excluding else_rule

    overlap_percentage = calculate_overlap_percentage(ruleset, X_test, cover_mat=cover_mat_test)

    overlap_analysis_res = overlap_analysis(ruleset, cover_mat_train)
    overlap_prob_diffs_mean, overlap_prob_diffs_max, modelling_group_coverage = \
        overlap_analysis_res["diff_probs_mean"], overlap_analysis_res["diff_probs_max"], overlap_analysis_res["modelling_group_counts"]

    cover_matrix_test_str = str(cover_mat_test)
    cover_matrix_test_str = cover_matrix_test_str.replace("\n", " ")

    cover_matrix_train_str = str(cover_mat_train)
    cover_matrix_train_str = cover_matrix_train_str.replace("\n", " ")


    random_picking_roc_auc, random_picking_pr_auc, random_picking_brier_score, random_picking_logloss = \
        calculate_random_picking_pred_performance(ruleset=ruleset, X=X_test, y=y_test, num_repetition=10, cover_mat=cover_mat_test)

    # overlap analysis
    overlap_prob_diff_analysis_res = overlap_prob_diff_analysis(ruleset, cover_mat_test, y_pred_prob=res)
    # overlap_prob_diff_analysis_res_DiffProbsMean = overlap_prob_diff_analysis_res["diff_probs_mean"]
    # overlap_prob_diff_analysis_res_DiffProbsMax = overlap_prob_diff_analysis_res["diff_probs_max"]
    # overlap_prob_diff_analysis_res_DiffProbsVar = overlap_prob_diff_analysis_res["diff_probs_var"]
    overlap_prob_diff_analysis_res_IndividualProbsAll = overlap_prob_diff_analysis_res["individual_probs_all"]

    overlap_prob_diff_analysis_res_train = overlap_prob_diff_analysis(ruleset, cover_mat_train, y_pred_prob=res_train)
    overlap_prob_diff_analysis_res_IndividualProbsAll_train = overlap_prob_diff_analysis_res_train["individual_probs_all"] 


    exp_res = {"roc_auc_test": roc_auc, "roc_auc_train": roc_auc_train,
               "data_name": data_name, "fold_index": fold, "nrow": X_train.shape[0], "ncol": X_train.shape[1],
               "num_rules": len(ruleset.rules), "avg_rule_length": avg_rule_length,
               "train_test_prob_diff": train_test_prob_diff,
               "pr_auc_train": pr_auc_train, "pr_auc_test": pr_auc_test,
               "Brier_train": Brier_train, "Brier_test": Brier_test,
               "logloss_train": logloss_train, "logloss_test": logloss_test,
               "runtime": end_time - start_time,
               "avg_num_literals_for_each_datapoint": avg_num_literals_for_each_datapoint,
               "overlap_perc": overlap_percentage, "random_picking_roc_auc": random_picking_roc_auc,
               "random_picking_pr_auc":random_picking_pr_auc, "random_picking_brier_score": random_picking_brier_score,
               "random_picking_logloss":random_picking_logloss,
               "accuracy_test": accuracy_test, "accuracy_train": accuracy_train,
               "rules_prob_test": rules_test_prob, "rules_prob_train": rules_train_prob,
               "cover_matrix_test": cover_mat_test, "cover_matrix_train": cover_mat_train,
               "cover_matrix_test_str": cover_matrix_test_str, "cover_matrix_train_str": cover_matrix_train_str,
               "overlap_prob_diffs_mean": overlap_prob_diffs_mean, "overlap_prob_diffs_max": overlap_prob_diffs_max,
               "modelling_group_coverage": modelling_group_coverage, 
               "y_pred_prob": res, "y_pred_prob_train": res_train}
    
    exp_report = exp_res.copy()
    # drop cover_matrix and rules_prob to reduce size
    del exp_report["cover_matrix_test"]
    del exp_report["cover_matrix_train"]
    del exp_report["cover_matrix_test_str"]
    del exp_report["cover_matrix_train_str"]
    del exp_report["rules_prob_test"]
    del exp_report["rules_prob_train"]
    # add mean diff of rule_prob_train and rule_prob_test
    # first get the predicted probs for each rule
    pred_class = np.argmax(rules_train_prob, axis=1)
    rule_probs_pred_class_train = np.array([rules_train_prob[i][pred_class[i]] for i in range(len(rules_train_prob))])
    rule_probs_pred_class_test = np.array([rules_test_prob[i][pred_class[i]] for i in range(len(rules_test_prob))])
    exp_report["mean_rule_probs_TrainTestDiff"] = np.mean(np.abs(rule_probs_pred_class_train - rule_probs_pred_class_test))
    # add weighted mean
    exp_report["weighted_mean_rule_probs_TrainTestDiff"] = np.average(np.abs(rule_probs_pred_class_train - rule_probs_pred_class_test), weights=[r.coverage for r in ruleset.rules] + [ruleset.else_rule_coverage])
    
    overlap_prob_diff_analysis_res_IndividualProbsAll_train_test = {"train": overlap_prob_diff_analysis_res_IndividualProbsAll_train,
                                                                    "test": overlap_prob_diff_analysis_res_IndividualProbsAll}

    return exp_res, exp_report, overlap_prob_diff_analysis_res_IndividualProbsAll_train_test


def overlap_prob_diff_analysis(ruleset, cover_mat, y_pred_prob):
    # for all data points, calculate the following: 
    # 1) all individual rules' prob. estimates (return as a numpy array)
    # 2) the variance of the prob. estimates
    # 3) the percentage of data points that are covered by more than one rule
    # 4) the max difference between the predicted prob (by taking the union) and the individual rules' prob estimates
    num_data_points = cover_mat.shape[0]
    diff_probs_mean = []
    diff_probs_max = []
    diff_probs_var = []
    individual_probs_all = []

    for i in range(num_data_points):
        covering_rules_indices = np.where(cover_mat[i, :] == 1)[0]
        if len(covering_rules_indices) <= 1:
            diff_probs_mean.append(0)
            diff_probs_max.append(0)
            diff_probs_var.append(0)
            if len(covering_rules_indices) == 1:
                # check if it is the else rule
                if covering_rules_indices[0] >= len(ruleset.rules):
                    # else rule
                    individual_probs_all.append([ruleset.else_rule_p])
                else:
                    individual_probs_all.append([ruleset.rules[covering_rules_indices[0]].prob])
            else:
                # use else rule
                individual_probs_all.append([ruleset.else_rule_p])
        else:
            individual_probs = []
            for r_idx in covering_rules_indices:
                individual_probs.append(ruleset.rules[r_idx].prob)
            individual_probs = np.array(individual_probs)  # shape: (num_covering_rules, num_class)
            individual_probs_all.append(individual_probs)
            
            predicted_class = np.argmax(y_pred_prob[i, :])
            indvidual_probs_on_predicted_class = individual_probs[:, predicted_class]
            
            individual_to_predicted_diff = np.abs(indvidual_probs_on_predicted_class - y_pred_prob[i, predicted_class])
            diff_mean = np.mean(individual_to_predicted_diff)
            diff_var = np.var(individual_to_predicted_diff)
            diff_max = np.max(individual_to_predicted_diff)

            diff_probs_mean.append(diff_mean)
            diff_probs_max.append(diff_max)
            diff_probs_var.append(diff_var)

    overlap_perc = np.sum(np.sum(cover_mat, axis=1) > 1) / num_data_points
    return {"diff_probs_mean": np.mean(diff_probs_mean),
            "diff_probs_max": np.mean(diff_probs_max),
            "diff_probs_var": np.mean(diff_probs_var),
            "individual_probs_all": individual_probs_all,
            "overlap_perc": overlap_perc,
            "modelling_group_counts": np.sum(np.sum(cover_mat, axis=1) > 1)}

