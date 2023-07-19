import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *


X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

beamwidth = 1
data_info = DataInfo(X=X, y=y, num_candidate_cuts=10, max_rule_length=10, feature_names=X.columns, beam_width=1)
ruleset = Ruleset(data_info=data_info)

ruleset.fit_rulelist(max_iter=1000)

X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

res = predict_rulelist(ruleset, X_test, y_test)

readable_ruleset = get_readable_rules(ruleset)

roc_auc = roc_auc_score(y_test, res[0][:, 1])
pr_curve = precision_recall_curve(y_test, res[0][:, 1])
print(auc(pr_curve[1], pr_curve[0]))

print(roc_auc)


for i, r in enumerate(ruleset.rules):
    print(r.prob_excl, res[1][i], r.coverage_excl)

train_res = predict_rulelist(ruleset, X, y)
training_roc_auc = roc_auc_score(y, train_res[0][:, 1])
print("training roc auc: ", training_roc_auc)
