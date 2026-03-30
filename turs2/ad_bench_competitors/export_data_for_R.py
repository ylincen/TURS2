import sys

import numpy as np

# sys.path.extend(['~/projects/TURS'])
sys.path.extend(['/Users/yanglincen/projects/TURS'])

import os
import copy
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from turs2.Ruleset import *
from turs2.DataEncoding import *

data_names = os.listdir("../../ADbench_datasets_Classical")
data_names_selected = []
for data_name in data_names:
    d = np.load("../../ADbench_datasets_Classical/" + data_name)
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

add_noise=True

nrows = []
ncols = []
prob_positive = []
data_names = []
for data_name in data_names_selected:
    print("Running on: ", data_name)
    d = np.load("../../ADbench_datasets_Classical/" + data_name)
    X = d["X"]
    y = d["y"]
    num_class = len(np.unique(y))

    nrows.append(X.shape[0])
    ncols.append(X.shape[1])
    prob_positive.append(np.mean(y == 1))
    data_names.append(data_name)
    pd_data_description = pd.DataFrame(
        {
            "data": data_names,
            "nrows":nrows,
            "ncols":ncols,
            "prob_positivve":prob_positive
        }
    )
    pd_data_description.to_csv("adbench_data_description.csv")

    skf = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=1)  # can also use sklearn.model_selection.StratifiedKFold
    if add_noise:
        np.random.seed(0)
        corrupted_indices = np.random.choice(np.arange(len(y)), size=int(0.2 * len(y)), replace=False)
        y[corrupted_indices] = (~y[corrupted_indices].astype(bool)).astype(int)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        pd_X_train = pd.DataFrame(X_train)
        pd_y_train = pd.DataFrame(y_train)
        pd_d_train = pd.concat([pd_X_train, pd_y_train], axis=1)

        pd_X_test, pd_y_test = pd.DataFrame(X_test), pd.DataFrame(y_test)
        pd_d_test = pd.concat([pd_X_test, pd_y_test], axis=1)

        pd_d_train.to_csv("noise20percent_train_test_split_data/" + data_name + "_train_fold_" + str(i), index=False)
        pd_d_test.to_csv("noise20percent_train_test_split_data/" + data_name + "_test_fold_" + str(i), index=False)