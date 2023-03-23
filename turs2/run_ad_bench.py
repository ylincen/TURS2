import sys
# sys.path.extend(['~/projects/TURS'])
sys.path.extend(['/Users/yanglincen/projects/TURS'])

import os
import copy
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

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

trial_run = True

time_all_data = []
auc_all_data = []
F1_score_all_data = []
PR_auc_all_data = []
data_name_all_data = []

for data_name in data_names_selected:
    print("Running on: ", data_name)
    d = np.load("../ADbench_datasets_Classical/" + data_name)
    X = d["X"]
    y = d["y"]
    if trial_run:
        rows = np.random.choice(np.arange(len(X)), size=min(1000, len(X)), replace=False)
        cols = np.random.choice(np.arange(X.shape[1]), size=min(5, X.shape[1]), replace=False)
        X = X[rows][:, cols]
        y = y[rows]
    num_class = len(np.unique(y))

    skf = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=1)  # can also use sklearn.model_selection.StratifiedKFold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        feature_names = ["X" + str(i) for i in range(X_train.shape[1])]
        data_info = DataInfo(X=X_train, y=y_train, num_candidate_cuts=20, max_rule_length=10,
                             feature_names=feature_names, beam_width=1)

        time0 = time.time()
        data_encoding = NMLencoding(data_info)
        model_encoding = ModelEncodingDependingOnData(data_info)
        ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

        ruleset.fit(max_iter=1000, printing=False)
        res = predict_ruleset(ruleset, X_test, y_test)

        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, res[0][:, 1])
        else:
            roc_auc = roc_auc_score(y_test, res[0], multi_class="ovr")
        runtime = time.time() - time0

        pr = precision_recall_curve(y_test, res[0][:, 1])  # all ad bench datasets are binary!
        pr_auc = auc(pr[1], pr[0])

        f1 = f1_score(y_test, res[0][:, 0] > res[0][:, 1])
        print("roc_auc: ", roc_auc)

        time_all_data.append(runtime)
        auc_all_data.append(roc_auc)
        F1_score_all_data.append(f1)
        PR_auc_all_data.append(pr_auc)
        data_name_all_data.append(data_name)

        res_pd = pd.DataFrame({
            "data": data_name_all_data,
            "time": time_all_data,
            "roc_auc": auc_all_data,
            "f1": F1_score_all_data,
            "pr_auc": PR_auc_all_data
        })

        res_pd.to_csv('res.csv')