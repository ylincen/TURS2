import copy

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.ensemble import RandomForestClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

# X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_no_scale.csv')
# y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

# X = X.iloc[:, :10]


beamwidth = 1
data_info = DataInfo(X=X, y=y, num_candidate_cuts=20, max_rule_length=5, feature_names=X.columns, beam_width=1)

data_encoding = NMLencoding(data_info)
model_encoding = ModelEncodingDependingOnData(data_info)
ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

ruleset.fit(max_iter=1000)


X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')
# X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_no_scale.csv')
# y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')


res = predict_ruleset(ruleset, X_test, y_test)

# readable = get_readable_rules(ruleset)

roc_auc = roc_auc_score(y_test, res[0][:, 1])
pr_curve = precision_recall_curve(y_test, res[0][:, 1])
print(auc(pr_curve[1], pr_curve[0]))

print(roc_auc)

covered = (res[0][:, 0] != ruleset.else_rule_p[0])
roc_auc_score(y_test[covered], res[0][covered, 1])

for i, r in enumerate(ruleset.rules):
    print(r.prob_excl, r.prob, res[1][i], r.coverage_excl, r.coverage)

rf = RandomForestClassifier()
rf.fit(X[ruleset.uncovered_bool], y[ruleset.uncovered_bool].to_numpy().flatten())
ypred_prob2 = rf.predict_proba(X_test[res[2]])

rf_full = RandomForestClassifier()
rf_full.fit(X, y.to_numpy().flatten())
ypred_prob_rf = rf_full.predict_proba(X_test)

pred_res = copy.deepcopy(res[0])
pred_res[res[2]] = ypred_prob2

roc_auc_score(y_test, pred_res[:,1])

roc_auc_score(y_test, ypred_prob_rf[:,1])