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
from tree_utils import *

data_path = "xml_challenge/heloc_dataset_v1.csv"
d = pd.read_csv(data_path)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
X = d.iloc[:, 1:].to_numpy()
y = d.iloc[:, 0].to_numpy()
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

X_train = dtrain.iloc[:, 1:].to_numpy()
y_train = dtrain.iloc[:, 0].to_numpy()

X_test = dtest.iloc[:, 1:].to_numpy()
y_test = dtest.iloc[:, 0].to_numpy()

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
roc_auc_score(y_test, clfs[best_clf_index].predict_proba(X_test)[:,1])

# Check how mdl score changes as the ccp_alpha grows:
# Turns out that the mdl_optimal ccp_alpha is 0: meaning that our regret is too small!!!
res = {}
for kk in range(max(0, best_clf_index - 30), min(best_clf_index + 30, len(clfs))):
    clf = clfs[kk]
    num_rules = clf.get_n_leaves()

    which_paths_train = clf.apply(X_train)
    train_membership = np.zeros((num_rules, X_train.shape[0]), dtype=bool)

    which_paths_train_dic = {}
    counter = 0
    for i in range(X_train.shape[0]):
        if which_paths_train[i] in which_paths_train_dic:
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
        else:
            which_paths_train_dic[which_paths_train[i]] = counter
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
            counter += 1

    counts_per_leaf = np.sum(train_membership, axis=1)
    sum_regrets = 0
    for count in counts_per_leaf:
        sum_regrets += regret(count, 2)

    sum_cl_data = 0  # this does NOT include regret!
    for train_membership_for_each_rule in train_membership:
        sum_cl_data += get_entropy(y_train[train_membership_for_each_rule], 2)
    res[kk] = sum_regrets + sum_cl_data


res2 = {}
res2_regrets = {}
res2_cl_data = {}
# for kk in range(max(0, best_clf_index - 30), min(best_clf_index + 30, len(clfs))):
for kk in range(len(clfs)):
    clf = clfs[kk]
    num_rules = clf.get_n_leaves()

    which_paths_train = clf.apply(X_train)
    train_membership = np.zeros((num_rules, X_train.shape[0]), dtype=bool)

    which_paths_train_dic = {}
    counter = 0
    for i in range(X_train.shape[0]):
        if which_paths_train[i] in which_paths_train_dic:
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
        else:
            which_paths_train_dic[which_paths_train[i]] = counter
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
            counter += 1
    sum_regrets = 0

    # counts_per_leaf = np.sum(train_membership, axis=1)
    # for count in counts_per_leaf:
    #     sum_regrets += regret(count, 2)
    # sum_regrets += regret(X_train.shape[0], num_rules)
    sum_regrets += regret(X_train.shape[0], 2 * num_rules)

    sum_cl_data = 0  # this does NOT include regret!
    for train_membership_for_each_rule in train_membership:
        sum_cl_data += get_entropy(y_train[train_membership_for_each_rule], 2)
    res2[kk] = sum_regrets + sum_cl_data
    res2_regrets[kk] = sum_regrets
    res2_cl_data[kk] = sum_cl_data

print(np.argmin(list(res2.values())))
print(np.argmin(list(res2_regrets.values())))
print(np.argmin(list(res2_cl_data.values())))

clfs[np.argmin(list(res2.values()))].score(X_test, y_test)
roc_auc_score(y_test, clfs[np.argmin(list(res2.values()))].predict_proba(X_test)[:,1])