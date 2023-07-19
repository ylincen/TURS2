import sys

# for mac local
sys.path.extend(['/Users/yanglincen/projects/TURS'])
sys.path.extend(['/Users/yanglincen/projects/TURS/turs2'])
# for DSlab server:
sys.path.extend(['/home/yangl3/projects/TURS'])
sys.path.extend(['/home/yangl3/projects/TURS/turs2'])

import numpy as np
import pandas as pd
import copy
import time
import cProfile
from datetime import datetime
from line_profiler import LineProfiler
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

datasets_without_header_row = ["chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine"]
datasets_with_header_row = ["avila", "anuran", "diabetes"]

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def _read_data(data_name):
    data_path = "../datasets/" + data_name + ".csv"
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
    if data_name == "anuran":
        d = d.iloc[:, 1:]
    return d

def preprocess_data(d):
    le = LabelEncoder()
    d.iloc[:, -1] = le.fit_transform(d.iloc[:, -1])

    le_feature = OneHotEncoder(sparse=False, dtype=int, drop="if_binary")

    for icol in range(d.shape[1] - 1):
        if d.iloc[:, icol].dtype == "float":
            d_transformed = d.iloc[:, icol]
        elif d.iloc[:, icol].dtype == "int" and len(np.unique(d.iloc[:, icol])) > 20:
            d_transformed = d.iloc[:, icol]
        else:
            d_transformed = le_feature.fit_transform(d.iloc[:, icol:(icol+1)])
            d_transformed = pd.DataFrame(d_transformed)
        if icol == 0:
            d_feature = d_transformed
        else:
            d_feature = pd.concat([d_feature, d_transformed], axis=1)

    d = pd.concat([d_feature, d.iloc[:, -1]], axis=1)
    d.columns = ["X" + str(i) for i in range(d.shape[1])]
    return d

def calculate_roc_auc_and_logloss(ruleset, y_test, y_pred_prob, y_train, y_pred_prob_train):
    if ruleset.data_info.num_class == 2:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        roc_auc_train = roc_auc_score(y_train, y_pred_prob_train[:, 1])

        logloss_train = log_loss(y_train, y_pred_prob_train[:, 1])
        logloss_test = log_loss(y_test, y_pred_prob[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", )
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
    return [pr_auc_test, pr_auc_train, Brier_test, Brier_train]

def calculate_train_test_prob_diff(ruleset, X_test, y_test):
    # weighted average of train_test prob. est. difference for each rule,
    rule_test_prob_info = get_rule_local_prediction_for_unseen_data(ruleset, X_test, y_test)
    rule_test_prob = rule_test_prob_info["rules_test_p"]
    rule_test_prob.append(rule_test_prob_info["else_rule_p"])
    rules_train_prob = [r.prob for r in ruleset.rules]
    rules_train_prob.append(ruleset.else_rule_p)
    rules_coverage_including_else = [r.coverage for r in ruleset.rules]
    rules_coverage_including_else.append(ruleset.else_rule_coverage)
    p_diff_rules = []
    for tr_p_, test_p_ in zip(rules_train_prob, rule_test_prob):
        p_diff = np.mean(abs(tr_p_ - test_p_))
        p_diff_rules.append(p_diff)
    train_test_prob_diff = np.average(p_diff_rules, weights=rules_coverage_including_else)
    return train_test_prob_diff

def calculate_random_picking_pred_performance(ruleset, X, y, num_repetition):
    random_picking_roc_auc = []
    random_picking_pr_auc = []
    random_picking_brier_score = []
    random_picking_logloss = []

    for _ in range(num_repetition):
        y_pred_prob = predict_random_picking_for_overlaps(ruleset=ruleset, X=X, seed=_)
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

    roc_auc, roc_auc_train, logloss_test, logloss_train = \
        calculate_roc_auc_and_logloss(ruleset=ruleset, y_test=y_test, y_train=y_train,
                                      y_pred_prob=res, y_pred_prob_train=res_train)

    pr_auc_test, pr_auc_train, Brier_test, Brier_train = \
        calculate_brier_and_prauc(ruleset=ruleset, y_test=y_test, y_train=y_train,
                                  y_pred_prob_train=res_train, y_pred_prob=res)

    avg_rule_length = calculate_rule_lengths(ruleset)

    train_test_prob_diff = calculate_train_test_prob_diff(ruleset, X_test, y_test)

    rules_p_value_permutations, rules_p_value_chi2 = significance_rules(ruleset, X_test, y_test, 500)
    insig_res_overlaps = insignificance_overlap_all(ruleset, X_test, y_test, 500)

    avg_num_literals_for_each_datapoint = explainability_analysis(ruleset, X_test)  # excluding else_rule

    overlap_percentage = calculate_overlap_percentage(ruleset, X_test)

    random_picking_roc_auc, random_picking_pr_auc, random_picking_brier_score, random_picking_logloss = \
        calculate_random_picking_pred_performance(ruleset=ruleset, X=X_train, y=y_test, num_repetition=10)

    exp_res = {"roc_auc_test": roc_auc, "roc_auc_train": roc_auc_train,
               "data_name": data_name, "fold_index": fold, "nrow": X_train.shape[0], "ncol": X_train.shape[1],
               "num_rules": len(ruleset.rules), "avg_rule_length": avg_rule_length,
               "train_test_prob_diff": train_test_prob_diff,
               "pr_auc_train": pr_auc_train, "pr_auc_test": pr_auc_test,
               "Brier_train": Brier_train, "Brier_test": Brier_test,
               "logloss_train": logloss_train, "logloss_test": logloss_test,
               "runtime": end_time - start_time,
               "rules_p_value_permutations_significance_perc": np.mean(rules_p_value_permutations < 0.05),
               "rules_p_value_permutations": rules_p_value_permutations,
               "overlap_significances_perc": 1 - insig_res_overlaps["insignificance_percentage"],
               "avg_num_literals_for_each_datapoint": avg_num_literals_for_each_datapoint,
               "overlap_perc": overlap_percentage, "random_picking_roc_auc": random_picking_roc_auc,
               "random_picking_pr_auc":random_picking_pr_auc, "random_picking_brier_score": random_picking_brier_score,
               "random_picking_logloss":random_picking_logloss}
    return exp_res

def run_(data_name, fold_given=None):
    d = _read_data(data_name)
    d = preprocess_data(d)

    X = d.iloc[:, :d.shape[1] - 1].to_numpy()
    y = d.iloc[:, d.shape[1] - 1].to_numpy()

    kf = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
    kfold = kf.split(X=X, y=y)
    kfold_list = list(kfold)

    print("running: ", data_name)
    for fold in range(5):
        if fold_given is not None and fold != fold_given:
            continue
        dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
        dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

        X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
        y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
        X_test = dtest.iloc[:, :-1].to_numpy()
        y_test = dtest.iloc[:, -1].to_numpy()

        start_time = time.time()
        data_info = DataInfo(X=X_train, y=y_train, beam_width=5)

        data_encoding = NMLencoding(data_info)
        model_encoding = ModelEncodingDependingOnData(data_info)
        ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
        ruleset.fit(max_iter=1000, printing=True)
        end_time = time.time()

        ## ROC_AUC and log-loss
        exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
        exp_res_alldata.append(exp_res)
    exp_res_df = pd.DataFrame(exp_res_alldata)
    if fold_given is None:
        res_file_name = "./" + date_and_time + "_" + data_name + "_uci_datasets_res.csv"
    else:
        res_file_name = "./" + date_and_time + "_" + data_name + "_fold" + str(fold_given) + "_uci_datasets_res.csv"
    exp_res_df.to_csv(res_file_name, index=False)