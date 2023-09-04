import sys

current_dir = os.getcwd()
sys.path.extend([current_dir + "/turs2/"])

# for mac local
# sys.path.extend(['/Users/yanglincen/projects/TURS'])
# sys.path.extend(['/Users/yanglincen/projects/TURS/turs2'])
# # for DSlab server:
# sys.path.extend(['/home/yangl3/projects/turs'])
# sys.path.extend(['/home/yangl3/projects/turs/turs2'])

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

np.seterr(all='raise')

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
datasets_without_header_row = ["chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                                   "magic", "car", "tic-tac-toe", "wine"]
datasets_with_header_row = ["avila", "anuran", "diabetes"]

datasets_with_header_row = datasets_with_header_row + ["Vehicle", "DryBeans"]
datasets_without_header_row = datasets_without_header_row + ["glass", "pendigits", "HeartCleveland"]

if len(sys.argv) == 3:
    data_name=sys.argv[1]
    fold_given=int(sys.argv[2])
elif len(sys.argv) == 2:
    data_name=sys.argv[1]
    fold_given=None
else:
    data_name = "iris"
    fold_given = 0

d = read_data(data_name, datasets_without_header_row=datasets_without_header_row,
              datasets_with_header_row=datasets_with_header_row)
d = preprocess_data(d)

X = d.iloc[:, :d.shape[1] - 1].to_numpy()
y = d.iloc[:, d.shape[1] - 1].to_numpy()

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=X, y=y)
kfold_list = list(kfold)

for fold in range(5):
    if fold_given is not None and fold != fold_given:
        continue
    print("running: ", data_name, "; fold: ", fold)
    dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
    dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

    X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
    y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
    X_test = dtest.iloc[:, :-1].to_numpy()
    y_test = dtest.iloc[:, -1].to_numpy()

    start_time = time.time()
    alg_config = AlgConfig(
        num_candidate_cuts=20, max_num_rules=500, max_grow_iter=500, num_class_as_given=None,
        beam_width=10,
        log_learning_process=False,
        dataset_name=None,
        feature_names=["X" + str(i) for i in range(X.shape[1])],
        validity_check="either"
    )
    data_info = DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=alg_config)

    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=1000, printing=True)
    end_time = time.time()

    ## ROC_AUC and log-loss
    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
    exp_res_alldata.append(exp_res)
exp_res_df = pd.DataFrame(exp_res_alldata)

folder_name = "exp_uci_" + date_and_time[:8]
os.makedirs(folder_name, exist_ok=True)
if fold_given is None:
    res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_uci_datasets_res.csv"
else:
    res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_fold" + str(fold_given) + "_uci_datasets_res.csv"
exp_res_df.to_csv(res_file_name, index=False)