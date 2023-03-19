import pickle

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.ensemble import RandomForestClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *

X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

for iter in range(1,8):
    print("iteration: ", iter)
    with open("ruleset" + str(iter) + ".pickle", "rb") as file:
        ruleset = pickle.load(file)

    res = predict_ruleset(ruleset, X_test, y_test)

    # get_readable_rules(ruleset)

    roc_auc = roc_auc_score(y_test, res[0][:, 1])
    pr_curve = precision_recall_curve(y_test, res[0][:, 1])
    print(auc(pr_curve[1], pr_curve[0]))

    print(roc_auc)

    # covered = (res[0][:, 0] != ruleset.else_rule_p[0])
    # roc_auc_score(y_test[covered], res[0][covered, 1])
    #
    # for i, r in enumerate(ruleset.rules):
    #     print(r.prob_excl, r.prob, res[1][i], r.coverage_excl, r.coverage)

# benchmark against random forest;
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X, y.to_numpy().reshape((1,-1))[0])
y_pred_prob = rf.predict_proba(X_test)

print(roc_auc_score(y_test.to_numpy().reshape((1,-1))[0], y_pred_prob[:,1]))
