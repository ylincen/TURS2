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

if len(sys.argv) == 1:
    data_names = os.listdir("../ADbench_datasets_Classical")
    data_names_selected = []
    for data_name in data_names:
        d = np.load("../ADbench_datasets_Classical/" + data_name)
        X = d["X"]
        y = d["y"]
        num_class = len(np.unique(y))
        # print(num_class)
        y_prob = calc_probs(y, num_class)
        if min(y_prob) > 0.05:
            continue
        elif len(y) > 1e5:
            continue
        else:
            data_names_selected.append(data_name)
    data_name = data_names_selected[0]
else:
    data_name = sys.argv[1]

print("Running on: ", data_name)
d_np = np.load("../ADbench_datasets_Classical/" + data_name)
X = d_np["X"]
y = d_np["y"]
d_original = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
d = preprocess_data(d_original)

num_class = len(np.unique(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    start_time = time.time()
    data_info = DataInfo(X=X_train, y=y_train, beam_width=20)

    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=1000, printing=True)
    end_time = time.time()
    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
    exp_res_alldata.append(exp_res)
exp_res_df = pd.DataFrame(exp_res_alldata)

folder_name = "exp_adbench_" + date_and_time[:8]
os.makedirs(folder_name, exist_ok=True)
res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_adbench_datasets_res.csv"

exp_res_df.to_csv(res_file_name, index=False)

