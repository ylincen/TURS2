################### NOTE: This will give error because at iteration 13, at some time, all feature values will be the same
### for some rule, and then no candidate cut points will be searched. Need a good way to solve this..

import numpy as np
import pandas as pd
import sklearn.datasets
import copy

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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

    data_info = DataInfo(X=X_train, y=y_train, num_candidate_cuts=10, max_rule_length=10, feature_names=dtrain.columns[:-1], beam_width=1)
    # ruleset = Ruleset(data_info=data_info)
    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

    ruleset.fit(max_iter=1000, printing=True)
    X_test = dtest.iloc[:, 1:].to_numpy()
    y_test = dtest.iloc[:, 0].to_numpy()
    res = predict_ruleset(ruleset, X_test, y_test)

    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, res[0][:, 1])
    else:
        roc_auc = roc_auc_score(y_test, res[0], multi_class="ovr")

    print("roc_auc: ", roc_auc)
    print("accuracy: ", np.mean(np.array(res[0][:, 1] > res[0][:, 0], dtype=int) == y_test))


#################### Benchmark against RF ##########################
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train.reshape((1,-1))[0])
y_pred_prob = rf.predict_proba(X_test)

print("RF ROC AUC: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("RF ACCURACY: ", np.mean(rf.predict(X_test) == y_test))