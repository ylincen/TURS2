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
from turs2.exp_predictive_perf import *



np.seterr(all='raise')

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")

not_use_excl_ = True

if len(sys.argv) == 1:
    data_name = "43_WDBC.npz.csv"
    folder_name = "../adbench_datasets_spurious_indep/"
    validity_check_ = "either"
    spurious_type = "indep"
else:
    data_name = sys.argv[1]
    validity_check_ = sys.argv[2]
    spurious_type = sys.argv[3]
    assert validity_check_ in ["either", "none"]
    assert spurious_type in ["indep", "dep"]

    if sys.platform == "darwin":
        folder_name = "../adbench_datasets_spurious/"
    else:
        # folder_name = "/data/yangl3/datasets_jmlr/"
        data_name = data_name + ".csv"
        folder_name = "/data/yangl3/datasets_jmlr/adbench_datasets_spurious_" + spurious_type + "/"


print("Running on: ", data_name)
# d_np = np.load(folder_name + data_name, + ".csv", allow_pickle=True)
data_path = folder_name + data_name
d_original = pd.read_csv(data_path)
# X = d_np["X"]
# y = d_np["y"]
# d_original = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
d = preprocess_data(d_original)

X, y = d.iloc[:, :d.shape[1] - 1].to_numpy(), d.iloc[:, d.shape[1] - 1].to_numpy()
num_class = len(np.unique(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    start_time = time.time()
    alg_config = AlgConfig(
        num_candidate_cuts=20, max_num_rules=500, max_grow_iter=500, num_class_as_given=None,
        beam_width=10,
        log_learning_process=False,
        dataset_name=None, X_test=None, y_test=None,
        rf_assist=False, rf_oob_decision_function=None,
        feature_names=["X" + str(i) for i in range(X.shape[1])],
        beamsearch_positive_gain_only=False, beamsearch_normalized_gain_must_increase_comparing_rulebase=False,
        beamsearch_stopping_when_best_normalized_gain_decrease=False,
        validity_check=validity_check_, rerun_on_invalid=False, rerun_positive_control=False, min_sample_each_rule=1
    )
    data_info = DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=alg_config,
                         not_use_excl_=not_use_excl_)

    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=1000, printing=True)
    end_time = time.time()
    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
    exp_res["not_use_excl_"] = not_use_excl_
    exp_res["validity_check_"] = validity_check_

    # check if the rules contain the spurious rules
    num_original_features = np.sum(["spurious" not in a for a in d.columns.tolist()]) - 1

    num_spurious_features_encountered = 0
    prob_diff_spurious_rules = []
    wts_train = []

    for ir, rule in enumerate(ruleset.rules):
        which_vars = np.where(rule.condition_count != 0)[0]
        num_spurious_features_encountered += np.sum(which_vars > num_original_features - 1)
        if len(which_vars) > 0:
            if np.sum(exp_res["rules_prob_train"][ir]) > 0.5:
                prob_diff = np.mean(abs(exp_res["rules_prob_test"][ir] - exp_res["rules_prob_train"][ir]))
                prob_diff_spurious_rules.append(prob_diff)
                wts_train.append(rule.coverage)
            else:
                # this means that no test point is covered by this rule
                pass
    if len(prob_diff_spurious_rules) > 0:
        exp_res["prob_diff_spurious_rules"] = np.mean(prob_diff_spurious_rules)
        exp_res["prob_diff_spurious_rules_weighted"] = np.average(prob_diff_spurious_rules, weights=wts_train)
    else:
        exp_res["prob_diff_spurious_rules"] = np.nan
        exp_res["prob_diff_spurious_rules_weighted"] = np.nan
    exp_res["num_spurious_features_encountered"] = num_spurious_features_encountered

    exp_res_alldata.append(exp_res)
exp_res_df = pd.DataFrame(exp_res_alldata)

folder_name = "NEW_adbench_spurious_" + spurious_type + "LOCALTEST_" + validity_check_ + date_and_time[:8]

os.makedirs(folder_name, exist_ok=True)
res_file_name = "./" + folder_name + "/" + data_name + ".csv"

exp_res_df.to_csv(res_file_name, index=False)

