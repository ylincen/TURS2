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
from turs2.exp_predictive_perf import *


def generate_data(nrow, ncol, seed):
    np.random.seed(seed)
    X = np.random.randint(2, size=(nrow, ncol)).astype(bool)
    X[:, 0] = np.random.choice([False, True], size=nrow, p=[0.8, 0.2])
    p1 = 0.7  # probability of Y = 0
    p0 = 0.9
    rule = X[:, 0]
    coverage = np.sum(rule)
    y = np.zeros(nrow, dtype=bool)
    y[rule] = np.random.choice([False, True], size=coverage, p=[p1, 1-p1])
    p_else = (p0 * nrow - p1 * coverage) / (nrow - coverage)
    y[~rule] = np.random.choice([False, True], size=nrow - coverage, p=[p_else, 1-p_else])
    X = X.astype(int)
    y = y.astype(int)
    return [X, y]

def run_turs(X_train, y_train, X_test, y_test, validity_check_option):
    start_time = time.time()
    alg_config = AlgConfig(
        num_candidate_cuts=20, max_num_rules=500, max_grow_iter=500, num_class_as_given=None,
        beam_width=10,
        log_learning_process=False,
        dataset_name=None, X_test=None, y_test=None,
        rf_assist=False, rf_oob_decision_function=None,
        feature_names=["X" + str(i) for i in range(X_train.shape[1])],
        beamsearch_positive_gain_only=False, beamsearch_normalized_gain_must_increase_comparing_rulebase=False,
        beamsearch_stopping_when_best_normalized_gain_decrease=False,
        validity_check=validity_check_option, rerun_on_invalid=False, rerun_positive_control=False
    )
    data_info = DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=alg_config)
    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=1000, printing=True)
    end_time = time.time()

    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, "simulation", 0, start_time, end_time)
    exp_res["total_cl"] = ruleset.total_cl
    exp_res["validity_check_option"] = validity_check_option
    return exp_res

exp_res_alliter = []
nrow = 10000
ncol = 100
for iter in range(100):
    print("iter", iter)

    for validity_check_option in ["no_check", "either"]:
        X_train, y_train = generate_data(nrow, ncol, seed=iter * 2)
        X_test, y_test = generate_data(nrow, ncol, seed=iter * 2 + 1)
        exp_res = run_turs(X_train, y_train, X_test, y_test, validity_check_option)
        exp_res_alliter.append(exp_res)
exp_res_df = pd.DataFrame(exp_res_alliter)

date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = "simulation_ablation_study" + date_and_time[:8]
os.makedirs(folder_name, exist_ok=True)
res_file_name = "./" + folder_name + "/" + date_and_time + "simulation_res.csv"
exp_res_df.to_csv(res_file_name, index=False)