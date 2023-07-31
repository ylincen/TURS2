import copy

import time
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, brier_score_loss, PrecisionRecallDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
# y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')
colnames = X.columns
# X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_StandardScaler_meanimputation_missing_features_dropped.csv')
# y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')


X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_no_scale.csv')
X = X.loc[:, colnames]
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')
X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_no_scale.csv')
X_test = X_test.loc[:, colnames]
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

ncol_original = X.shape[1]

# make it faster
# col_used = np.array([2,  19,  29,  30,  75, 212, 242, 275, 314, 319, 330], dtype=int)
# col_used = np.array([ 2,   8,  29,  75,  87, 140, 180, 186, 212, 266, 275, 313, 314, 319, 330], dtype=int)
# col_used = np.array([29,  60,  75,  81,  93, 115, 122, 131, 144, 185, 212, 242, 271, 324, 351, 400], dtype=int)
# col_used = np.unique(
#     np.array([2,  19,  29,  30,  75, 212, 242, 275, 314, 319, 330] +
#              [2,   8,  29,  75,  87, 140, 180, 186, 212, 266, 275, 313, 314, 319, 330] +
#              [29,  60,  75,  81,  93, 115, 122, 131, 144, 185, 212, 242, 271, 324, 351, 400] +
#              [69,  75, 212, 242, 319, 400], dtype=int)
# )
#
# # # col_used = np.array([69,  75, 212, 242, 319, 400], dtype=int)
#
# X = X.iloc[:, col_used]
# X_test = X_test.iloc[:, col_used]

print("===================Random Forest======================")
rf = RandomForestClassifier(n_estimators=200, n_jobs=5, oob_score=True, min_samples_leaf=30, max_features=10)
rf.fit(X, y.to_numpy().ravel())
# hyper_space = {"max_features": [10,20,30,40,50,70,100], "min_samples_leaf": [10,20,30,40]}
# rf = RandomForestClassifier()
# grid_cv = GridSearchCV(rf, hyper_space)
# grid_cv.fit(X, y.to_numpy().ravel())

print("===================Rule Learning======================")

beamwidth = 1
data_info = DataInfo(X=X, y=y, beam_width=5)

data_encoding = NMLencoding(data_info)
model_encoding = ModelEncodingDependingOnData(data_info, given_ncol=ncol_original)
ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

t0 = time.time()
ruleset.fit(max_iter=1000)
res = predict_ruleset(ruleset, X_test, y_test)
t1 = time.time() - t0
print("runtime", t1)
# readable = get_readable_rules(ruleset)

roc_auc = roc_auc_score(y_test, res[:, 1])
pr_curve = precision_recall_curve(y_test, res[:, 1])
print(auc(pr_curve[1], pr_curve[0]))

print("roc_auc on test data: ", roc_auc)
print("precision-recall auc on test data: ", auc(pr_curve[1], pr_curve[0]))
print("Brier score on test data: ", brier_score_loss(y_test, res[:, 1]))

train_res = predict_ruleset(ruleset, X, y)
roc_auc_train = roc_auc_score(y, train_res[:, 1])
pr_curve_train = precision_recall_curve(y, train_res[:, 1])

print("roc_auc on training data: ", roc_auc_train)
print("precision-recall auc on training data: ", auc(pr_curve_train[1], pr_curve_train[0]))
print("Brier score on train data: ", brier_score_loss(y, train_res[:, 1]))


print("===================Random Forest======================")

y_pred_p = rf.predict_proba(X_test)
y_pred = rf.predict(X_test)

y_pred_p_train = rf.predict_proba(X)
print("training set roc auc: ", roc_auc_score(y, y_pred_p_train[:,1]))
print("roc auc on test data: ", roc_auc_score(y_test, y_pred_p[:, 1]))
print("Brier score, train/test: ", brier_score_loss(y, y_pred_p_train[:, 1]), brier_score_loss(y_test, y_pred_p[:, 1]))
print("oob roc auc: ", roc_auc_score(y, rf.oob_decision_function_[:,1]))

else_rule_cover = ruleset.uncovered_indices
oob_df_else_rule = np.array(rf.oob_decision_function_)[:,1]
oob_df_else_rule[else_rule_cover] = np.median(oob_df_else_rule[else_rule_cover])
print("oob roc auc after putting else rule in: ", roc_auc_score(y, oob_df_else_rule))
pr_curve_squz = precision_recall_curve(y, oob_df_else_rule)
print("oob PR auc after squeezing all rules including the else: ", auc(pr_curve_squz[1], pr_curve_squz[0]))



col_used = np.zeros(X.shape[1], dtype=bool)
for r in ruleset.rules:
    col_used[np.where(r.condition_count != 0)[0]] = True

else_rule_cover = ruleset.uncovered_indices
oob_rules = np.array(rf.oob_decision_function_)[:,1]
oob_rules[else_rule_cover] = np.median(rf.oob_decision_function_[else_rule_cover, 1])
for mg in ruleset.modelling_groups:
    oob_rules[mg.bool_cover] = np.median(rf.oob_decision_function_[mg.bool_model, 1])
print("oob roc auc after squeezing all rules including the else: ", roc_auc_score(y, oob_rules))
pr_curve_squz = precision_recall_curve(y, oob_rules)
print("oob PR auc after squeezing all rules including the else: ", auc(pr_curve_squz[1], pr_curve_squz[0]))


pr_curve_rf = precision_recall_curve(y_test, y_pred_p[:,1])
print("RF precision-recall auc on test data: ", auc(pr_curve_rf[1], pr_curve_rf[0]))

pr_curve_rf_train = precision_recall_curve(y, y_pred_p_train[:,1])
print("RF precision-recall auc on train data: ", auc(pr_curve_rf_train[1], pr_curve_rf_train[0]))

analyze_res1 = False  # pruning ruleset
analyze_res2 = True   # check overlap of rules

### Analysis results
if analyze_res1:
    ruleset_pruned = copy.deepcopy(ruleset)
    for i in range(len(ruleset.rules),0, -1):
        ruleset_pruned = ruleset_pruned.ruleset_after_deleting_a_rule(i)

        res = predict_ruleset(ruleset_pruned, X_test, y_test)
        train_res = predict_ruleset(ruleset_pruned, X, y)

        roc_auc = roc_auc_score(y_test, res[:, 1])
        pr_curve = precision_recall_curve(y_test, res[:, 1])

        roc_auc_train = roc_auc_score(y, train_res[:, 1])
        pr_curve_train = precision_recall_curve(y, train_res[:, 1])

        print("Deleting ", i, "rules from last, precision-recall auc on training data: ", auc(pr_curve_train[1], pr_curve_train[0]))
        print("Deleting ", i, "rules from last, precision-recall auc on testing data: ", auc(pr_curve[1], pr_curve[0]))

if analyze_res2:
    res = []
    for mg in ruleset.modelling_groups:
        res1 = [mg.rules_involvde, mg.cover_count, tuple([round(ruleset.rules[ii].prob[1], ndigits=3) for ii in mg.rules_involvde])]
        res.append(res1)
    with pd.option_context('display.max_colwidth', None):
        print(pd.DataFrame(res))

