from sklearn import tree
import numpy as np
from nml_regret import *
from utils import calc_probs
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc, average_precision_score, f1_score, confusion_matrix
import copy
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data_path = "datasets/iris.csv"
d = pd.read_csv(data_path)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
X = d.iloc[:, :d.shape[1]-1].to_numpy()
y = d.iloc[:, d.shape[1]-1].to_numpy()
kfold = kf.split(X=X, y=y)
kfold_list = list(kfold)

fold = 0

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

X_train = dtrain.iloc[:, :d.shape[1]-1].to_numpy()
y_train = dtrain.iloc[:, d.shape[1]-1].to_numpy()

X_test = dtest.iloc[:, :d.shape[1]-1].to_numpy()
y_test = dtest.iloc[:, d.shape[1]-1].to_numpy()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

path = clf.cost_complexity_pruning_path(X_train, y_train)

clfs = []
for ccp_alpha in path["ccp_alphas"]:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

test_scores = [clf.score(X_test, y_test) for clf in clfs]
clf_best = clfs[np.argmax(test_scores)]

best_clf_index = np.argmax(test_scores)
roc_auc_score(y_test, clfs[best_clf_index].predict_proba(X_test), multi_class="ovr")

tree.plot_tree(clf_best)
plt.show()