import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from DataInfo import *
from newRuleset import *
from utils_pred import *
from sklearn.metrics import roc_auc_score


X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

d = X
d["readmission_label"] = y

d_small = d.iloc[:1000, :]
beamwidth = 1
data_info = DataInfo(data=d_small, max_bin_num=20, max_cat_level=1)
ruleset = Ruleset(data_info=data_info, features=data_info.features, target=data_info.target, number_of_rules_return=beamwidth,
                  number_of_init_rules=beamwidth)

# ruleset.build(max_iter=1000, beam_width=beamwidth, candidate_cuts=data_info.candidate_cuts, dump=True, print_or_not=True)
ruleset.build_rule_list(max_iter=1000, beam_width=beamwidth, candidate_cuts=data_info.candidate_cuts)

X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')
test_p = get_test_p_rulelist(rulelist=ruleset, X=X_test.to_numpy())

roc_auc = roc_auc_score(y_test, test_p[:, 1])
print("roc_auc: ", roc_auc)

for r in ruleset.rules:
    print(r.coverage_excl, r.prob_excl)

for r in ruleset.rules:
    print(r.cl_model, r.regret_excl)
