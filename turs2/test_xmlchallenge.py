################### NOTE: This will give error because at iteration 13, at some time, all feature values will be the same
### for some rule, and then no candidate cut points will be searched. Need a good way to solve this..

import numpy as np
import pandas as pd
import sklearn.datasets
import copy

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *

from turs2.ModelEncoding import *
from turs2.DataEncoding import *

data_path = "../xml_challenge/heloc_dataset_v1.csv"
print("Running TURS on: " + data_path)

d = pd.read_csv(data_path)

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
X = d.iloc[:, 1:].to_numpy()
y = d.iloc[:, 0].to_numpy()
kfold = kf.split(X=X, y=y)

kfold_list = list(kfold)

for fold in range(1):
    dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
    dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

    le = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
    for icol, tp in enumerate(dtrain.dtypes):
        if tp != float:
            feature_ = dtrain.iloc[:, icol].to_numpy()
            if len(np.unique(feature_)) > 5:
                continue
            feature_ = feature_.reshape(-1, 1)

            feature_test = dtest.iloc[:, icol].to_numpy()
            feature_test = feature_test.reshape(-1, 1)

            le.fit(feature_)
            dtrain.iloc[:, icol] = le.transform(feature_).reshape(1, -1)[0]
            dtest.iloc[:, icol] = le.transform(feature_test).reshape(1, -1)[0]

    X_train = dtrain.iloc[:, 1:dtrain.shape[1]].to_numpy()
    y_train = dtrain.iloc[:, 0].to_numpy()
    X_test = dtest.iloc[:, 1:].to_numpy()
    y_test = dtest.iloc[:, 0].to_numpy()

    # RF assisted
    rf = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True, min_samples_leaf=20)
    rf.fit(X_train, y_train.reshape((1, -1))[0])
    y_pred_prob = rf.predict_proba(X_test)
    oob_dec = rf.oob_decision_function_

    data_info = DataInfo(X=X_train, y=y_train, num_candidate_cuts=100, max_rule_length=20,
                         feature_names=dtrain.columns[:-1], beam_width=10,
                         X_test=X_test, y_test=y_test, log_learning_process=True, rf_oob_decision_=None)
    # ruleset = Ruleset(data_info=data_info)
    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

    ruleset.fit(max_iter=1000, printing=True)

    res = predict_ruleset(ruleset, X_test, y_test)

    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, res[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, res, multi_class="ovr")

    print("roc_auc: ", roc_auc)
    print("accuracy: ", np.mean(np.array(res[:, 1] > res[:, 0], dtype=int) == y_test))
    print("brier score: ", brier_score_loss(y_test, res[:, 1]))
    print("log loss: ", log_loss(y_test, res[:, 1]))


#################### Benchmark against RF ##########################

print("\nRF ROC AUC: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("RF ACCURACY: ", np.mean(rf.predict(X_test) == y_test))
print("RF brier score: ", brier_score_loss(y_test, y_pred_prob[:, 1]))
print("RF log loss: ", log_loss(y_test, y_pred_prob[:, 1]))


################### benchmark against fine-tuned decision tree ########
# tree = DecisionTreeClassifier(random_state=42)
#
# # Define grid of hyperparameters to search through
# param_grid = {
#     'ccp_alpha': np.linspace(0, 0.1, 50),  # cost-complexity pruning parameter
# }
#
# grid_search = GridSearchCV(tree, param_grid, cv=5)
# grid_search.fit(X_train, y_train)
#
# best_ccp_alpha = grid_search.best_params_['ccp_alpha']
# best_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=best_ccp_alpha)
# best_tree.fit(X_train, y_train)
#
# cart_pred_prob = best_tree.predict_proba(X_test)
# print("\nCART ROC AUC: ", roc_auc_score(y_test, cart_pred_prob[:,1]))
# print("CART ACCURACY: ", np.mean(best_tree.predict(X_test) == y_test))
# print("CART brier score: ", brier_score_loss(y_test, cart_pred_prob[:, 1]))
# print("CART log loss: ", log_loss(y_test, cart_pred_prob[:, 1]))