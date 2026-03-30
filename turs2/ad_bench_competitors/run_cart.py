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

trial_run = False

time_all_data = []
auc_all_data = []
F1_score_all_data = []
PR_auc_all_data = []
data_name_all_data = []

for data_name in data_names_selected:
    print("Running on: ", data_name)
    d = np.load("../../ADbench_datasets_Classical/" + data_name)
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

        time0 = time.time()
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.25, stratify=y_train)
        clf = DecisionTreeClassifier(random_state=0)
        path = clf.cost_complexity_pruning_path(X_tr, y_tr)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        clfs = []
        roc_auc_vals = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_tr, y_tr)
            yprob_pred = clf.predict_proba(X_val)
            roc_auc_vals.append(roc_auc_score(y_val, yprob_pred[:, 1]))

        ccp_alpha_best = ccp_alphas[np.argmax(roc_auc_vals)]
        model = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha_best)
        model.fit(X_tr, y_tr)
        y_pred_for_test = model.predict_proba(X_test)

        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_for_test[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_for_test, multi_class="ovr")
        runtime = time.time() - time0

        pr = precision_recall_curve(y_test, y_pred_for_test[:, 1])  # all ad bench datasets are binary!
        pr_auc = auc(pr[1], pr[0])

        f1 = f1_score(y_test, y_pred_for_test[:, 1] > y_pred_for_test[:, 0])
        print("roc_auc: ", roc_auc, " F1 score: ", f1)

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
            "pr_auc": PR_auc_all_data,
            "alg": "CART_cv"
        })

        res_pd.to_csv('res_CART_withCV.csv')