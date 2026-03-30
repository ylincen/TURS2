import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, log_loss

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *
from turs2.exp_utils import *


def read_data(data_name, datasets_without_header_row, datasets_with_header_row, folder_name="./datasets/"):
    data_path = folder_name + data_name + ".csv"
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        raise ValueError(f"'{data_name}' not found in either dataset list.")
    if data_name == "anuran":  # first column is a row index in the raw file
        d = d.iloc[:, 1:]
    return d


def preprocess_data(d, colnames=None, threshold_categorical=20):
    le = LabelEncoder()
    d = d.copy()
    last_col = d.columns[-1]
    d[last_col] = le.fit_transform(d.iloc[:, -1]).astype(int)

    if colnames is not None:
        new_colnames = []

    le_feature = OneHotEncoder(sparse_output=False, dtype=int, drop="if_binary")

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
            d_transformed = le_feature.fit_transform(d.iloc[:, icol:(icol + 1)])
            d_transformed = pd.DataFrame(d_transformed)
            if colnames is not None:
                new_colnames.extend([colnames[icol] + "_" + str(u) for u in np.unique(d.iloc[:, icol])])
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
        roc_auc       = roc_auc_score(y_test,  y_pred_prob[:, 1])
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train[:, 1])
        logloss_test  = log_loss(y_test,  y_pred_prob[:, 1])
        logloss_train = log_loss(y_train, y_pred_prob_train[:, 1])
    else:
        roc_auc       = roc_auc_score(y_test,  y_pred_prob,       multi_class="ovr")
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train, multi_class="ovr")
        logloss_test  = log_loss(y_test,  y_pred_prob)
        logloss_train = log_loss(y_train, y_pred_prob_train)
    return [roc_auc, roc_auc_train, logloss_test, logloss_train]


def calculate_rule_lengths(ruleset):
    rule_lengths = [np.count_nonzero(r.condition_count) for r in ruleset.rules]
    return np.mean(rule_lengths)


def calculate_brier_and_prauc(ruleset, y_train, y_test, y_pred_prob, y_pred_prob_train):
    Brier_train, Brier_test = 0, 0
    pr_auc_train, pr_auc_test = 0, 0

    # For binary classification use only class 1; for multi-class iterate over all classes.
    y_unique = np.arange(ruleset.data_info.num_class)
    if len(y_unique) == 2:
        y_unique = np.array([1], dtype=int)

    for yy in y_unique:
        positive_mask_train = (y_train == yy)
        positive_mask_test  = (y_test  == yy)

        Brier_train += np.sum((y_pred_prob_train[:, yy] - positive_mask_train) ** 2)
        Brier_test  += np.sum((y_pred_prob[:, yy]       - positive_mask_test)  ** 2)

        pr_train = precision_recall_curve(positive_mask_train, y_pred_prob_train[:, yy])
        pr_test  = precision_recall_curve(positive_mask_test,  y_pred_prob[:, yy])
        pr_auc_train += auc(pr_train[1], pr_train[0])
        pr_auc_test  += auc(pr_test[1],  pr_test[0])

    # Divide once after accumulating over all classes
    Brier_train /= len(y_pred_prob_train)
    Brier_test  /= len(y_pred_prob)

    return [pr_auc_test / len(y_unique), pr_auc_train / len(y_unique), Brier_test, Brier_train]


def calculate_train_test_prob_diff(ruleset, X_test, y_test, return_for_all_rules=False):
    rule_test_prob_info = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rules_test_prob  = rule_test_prob_info["rules_test_p"] + [rule_test_prob_info["else_rule_p"]]
    rules_train_prob = [r.prob for r in ruleset.rules] + [ruleset.else_rule_p]
    rules_coverage   = [r.coverage for r in ruleset.rules] + [ruleset.else_rule_coverage]

    p_diff_rules = [np.mean(abs(tr - te)) for tr, te in zip(rules_train_prob, rules_test_prob)]
    train_test_prob_diff = np.average(p_diff_rules, weights=rules_coverage)

    if return_for_all_rules:
        return [p_diff_rules, train_test_prob_diff, rules_test_prob, rules_train_prob]
    else:
        return train_test_prob_diff


def calculate_random_picking_pred_performance(ruleset, X, y, num_repetition, cover_mat):
    roc_aucs, pr_aucs, briers, loglosses = [], [], [], []
    for seed in range(num_repetition):
        y_pred_prob = predict_random_picking_for_overlaps(ruleset=ruleset, X=X, seed=seed, cover_mat=cover_mat)
        roc_auc, _, logloss_test, _ = calculate_roc_auc_and_logloss(
            ruleset=ruleset, y_test=y, y_pred_prob=y_pred_prob, y_train=y, y_pred_prob_train=y_pred_prob)
        pr_auc_test, _, brier_test, _ = calculate_brier_and_prauc(
            ruleset=ruleset, y_train=y, y_test=y, y_pred_prob=y_pred_prob, y_pred_prob_train=y_pred_prob)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc_test)
        briers.append(brier_test)
        loglosses.append(logloss_test)
    return [np.mean(roc_aucs), np.mean(pr_aucs), np.mean(briers), np.mean(loglosses)]


def calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time):
    res       = predict_ruleset(ruleset, X_test,  y_test)
    res_train = predict_ruleset(ruleset, X_train, y_train)
    cover_mat_test  = cover_matrix_fun(ruleset, X_test)
    cover_mat_train = cover_matrix_fun(ruleset, X_train)

    roc_auc, roc_auc_train, logloss_test, logloss_train = calculate_roc_auc_and_logloss(
        ruleset=ruleset, y_test=y_test, y_train=y_train, y_pred_prob=res, y_pred_prob_train=res_train)

    pr_auc_test, pr_auc_train, Brier_test, Brier_train = calculate_brier_and_prauc(
        ruleset=ruleset, y_test=y_test, y_train=y_train, y_pred_prob=res, y_pred_prob_train=res_train)

    accuracy_test  = np.mean(np.argmax(res,       axis=1) == y_test)
    accuracy_train = np.mean(np.argmax(res_train, axis=1) == y_train)

    avg_rule_length = calculate_rule_lengths(ruleset)

    _, train_test_prob_diff, rules_test_prob, rules_train_prob = \
        calculate_train_test_prob_diff(ruleset, X_test, y_test, return_for_all_rules=True)

    avg_num_literals = explainability_analysis(ruleset, X_test, cover_mat_test)
    overlap_percentage = calculate_overlap_percentage(ruleset, X_test, cover_mat=cover_mat_test)

    overlap_analysis_res = overlap_analysis(ruleset, cover_mat_train)
    overlap_prob_diffs_mean = overlap_analysis_res["diff_probs_mean"]
    overlap_prob_diffs_max  = overlap_analysis_res["diff_probs_max"]
    modelling_group_coverage = overlap_analysis_res["modelling_group_counts"]

    random_picking_roc_auc, random_picking_pr_auc, random_picking_brier_score, random_picking_logloss = \
        calculate_random_picking_pred_performance(ruleset=ruleset, X=X_test, y=y_test,
                                                  num_repetition=10, cover_mat=cover_mat_test)

    overlap_prob_diff_res_test  = overlap_prob_diff_analysis(ruleset, cover_mat_test,  y_pred_prob=res)
    overlap_prob_diff_res_train = overlap_prob_diff_analysis(ruleset, cover_mat_train, y_pred_prob=res_train)
    overlap_prob_diff_analysis_res_IndividualProbsAll_train_test = {
        "test":  overlap_prob_diff_res_test["individual_probs_all"],
        "train": overlap_prob_diff_res_train["individual_probs_all"],
    }

    exp_res = {
        "roc_auc_test": roc_auc, "roc_auc_train": roc_auc_train,
        "data_name": data_name, "fold_index": fold,
        "nrow": X_train.shape[0], "ncol": X_train.shape[1],
        "num_rules": len(ruleset.rules), "avg_rule_length": avg_rule_length,
        "train_test_prob_diff": train_test_prob_diff,
        "pr_auc_train": pr_auc_train, "pr_auc_test": pr_auc_test,
        "Brier_train": Brier_train, "Brier_test": Brier_test,
        "logloss_train": logloss_train, "logloss_test": logloss_test,
        "runtime": end_time - start_time,
        "avg_num_literals_for_each_datapoint": avg_num_literals,
        "overlap_perc": overlap_percentage,
        "random_picking_roc_auc": random_picking_roc_auc,
        "random_picking_pr_auc": random_picking_pr_auc,
        "random_picking_brier_score": random_picking_brier_score,
        "random_picking_logloss": random_picking_logloss,
        "accuracy_test": accuracy_test, "accuracy_train": accuracy_train,
        "rules_prob_test": rules_test_prob, "rules_prob_train": rules_train_prob,
        "cover_matrix_test": cover_mat_test, "cover_matrix_train": cover_mat_train,
        "overlap_prob_diffs_mean": overlap_prob_diffs_mean,
        "overlap_prob_diffs_max": overlap_prob_diffs_max,
        "modelling_group_coverage": modelling_group_coverage,
        "y_pred_prob": res, "y_pred_prob_train": res_train,
    }

    exp_report = {k: v for k, v in exp_res.items()
                  if k not in ("cover_matrix_test", "cover_matrix_train",
                               "rules_prob_test", "rules_prob_train")}

    pred_class = np.argmax(rules_train_prob, axis=1)
    rule_probs_pred_class_train = np.array([rules_train_prob[i][pred_class[i]] for i in range(len(rules_train_prob))])
    rule_probs_pred_class_test  = np.array([rules_test_prob[i][pred_class[i]]  for i in range(len(rules_test_prob))])
    abs_diff = np.abs(rule_probs_pred_class_train - rule_probs_pred_class_test)
    weights  = [r.coverage for r in ruleset.rules] + [ruleset.else_rule_coverage]
    exp_report["mean_rule_probs_TrainTestDiff"]          = np.mean(abs_diff)
    exp_report["weighted_mean_rule_probs_TrainTestDiff"] = np.average(abs_diff, weights=weights)

    return exp_res, exp_report, overlap_prob_diff_analysis_res_IndividualProbsAll_train_test


def overlap_prob_diff_analysis(ruleset, cover_mat, y_pred_prob):
    num_data_points = cover_mat.shape[0]
    diff_probs_mean, diff_probs_max, diff_probs_var = [], [], []
    individual_probs_all = []

    for i in range(num_data_points):
        covering_rules_indices = np.where(cover_mat[i, :] == 1)[0]
        if len(covering_rules_indices) <= 1:
            diff_probs_mean.append(0)
            diff_probs_max.append(0)
            diff_probs_var.append(0)
            if len(covering_rules_indices) == 1:
                if covering_rules_indices[0] >= len(ruleset.rules):
                    individual_probs_all.append([ruleset.else_rule_p])
                else:
                    individual_probs_all.append([ruleset.rules[covering_rules_indices[0]].prob])
            else:
                individual_probs_all.append([ruleset.else_rule_p])
        else:
            individual_probs = np.array([ruleset.rules[r_idx].prob for r_idx in covering_rules_indices])
            individual_probs_all.append(individual_probs)

            predicted_class = np.argmax(y_pred_prob[i, :])
            individual_on_predicted = individual_probs[:, predicted_class]
            diffs = np.abs(individual_on_predicted - y_pred_prob[i, predicted_class])

            diff_probs_mean.append(np.mean(diffs))
            diff_probs_max.append(np.max(diffs))
            diff_probs_var.append(np.var(diffs))

    return {
        "diff_probs_mean": np.mean(diff_probs_mean),
        "diff_probs_max": np.mean(diff_probs_max),
        "diff_probs_var": np.mean(diff_probs_var),
        "individual_probs_all": individual_probs_all,
        "overlap_perc": np.sum(np.sum(cover_mat, axis=1) > 1) / num_data_points,
        "modelling_group_counts": np.sum(np.sum(cover_mat, axis=1) > 1),
    }
