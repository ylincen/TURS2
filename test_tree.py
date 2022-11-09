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
from tree_utils import *
from scipy.special import gammaln


data_path = "datasets/diabetes.csv"
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

# choose the one with the largest ccp_alpha when the scores have a tie;
best_clf_index = len(test_scores) - np.argmax(test_scores[::-1]) - 1
clf_best = clfs[best_clf_index]

num_class = len(np.unique(y))
if num_class != 2:
    auc = roc_auc_score(y_test, clfs[best_clf_index].predict_proba(X_test), multi_class="ovr")
else:
    auc = roc_auc_score(y_test, clfs[best_clf_index].predict_proba(X_test)[:, 1])

aucs = []
for clf in clfs:
    if num_class != 2:
        aucs.append( roc_auc_score(y_test, clf.predict_proba(X_test), multi_class="ovr") )
    else:
        aucs.append( roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) )


# tree.plot_tree(clf_best)
# plt.show()

# calculate the MDL_score
## fixed_setting NML
def get_negloglike(X, y, clf):
    num_class = len(np.unique(y))
    which_nodes = clf.apply(X)

    whether_leaf_nodes = (clf.tree_.children_left == -1)
    negloglike = []
    for i in np.unique(which_nodes):
        instances_covered = (which_nodes == i)
        y_covered = y[instances_covered]

        p = calc_probs(y_covered, num_class)
        p = p[p != 0]

        negloglike.append(-np.sum(np.log2(p) * p) * np.count_nonzero(instances_covered))
    return np.sum(negloglike)

def get_counts_per_leaf(X, clf):
    which_nodes = clf.apply(X)
    whether_leaf_nodes = (clf.tree_.children_left == -1)
    counts = np.bincount(which_nodes)
    counts = counts[whether_leaf_nodes]
    return counts

counts_list = []
for i, clf in enumerate(clfs):
    counts_list.append(get_counts_per_leaf(X_train, clf))


def fixed_setting_nml(counts, num_class):
    regt = 0
    for count in counts:
        regt += regret(count, num_class)

    return regt


def random_setting_nml_my_guess(counts, num_class):
    total_sample = np.sum(counts)
    regt = regret(total_sample, num_class * len(counts)) - regret(total_sample, len(counts))

    return regt


def chooselog2(N, k):
      return (gammaln(N+1) - gammaln(N-k+1) - gammaln(k+1)) / log(2)


def get_cl_model(tree, X_train):
    features = tree.feature
    cuts = tree.threshold
    is_leaf = (tree.children_left == -1)
    n_leaves = np.count_nonzero(is_leaf)

    # Assume that we always go left if we can, the bits we need are:
    bits_for_number_of_nodes = 2 * np.log2(len(is_leaf))
    bits_for_leaves_position = chooselog2(len(is_leaf), n_leaves)  ## To MUCH REDUNDANCY HERE;

    num_features = len(X_train[0])

    features_non_leaf = features[~is_leaf]
    p_feature_name = calc_probs(features_non_leaf, num_features)

    naive_bits_for_which_dimension_in_condition = np.log2(num_features) * len(features_non_leaf)
    nml_bits_for_which_dimension_in_condition = \
        -np.sum(np.log2(p_feature_name[p_feature_name != 0]) * p_feature_name[p_feature_name != 0]) + \
        regret(len(features_non_leaf), num_features)

    bits_for_cuts = np.sum(np.log2(tree.n_node_samples[tree.n_node_samples != 0]))

    cl_model = bits_for_number_of_nodes + bits_for_leaves_position + nml_bits_for_which_dimension_in_condition + bits_for_cuts

    return cl_model

nml_naive, nml_myguess, cl_model, negloglike = [], [], [], []
num_class = len(np.unique(y))
for clf in clfs:
    tree = clf.tree_
    cl_model.append(get_cl_model(tree, X_train))

    counts = get_counts_per_leaf(X_train, clf)
    nml_naive.append(fixed_setting_nml(counts, num_class))
    nml_myguess.append(random_setting_nml_my_guess(counts, num_class))
    negloglike.append(get_negloglike(X_train, y_train, clf))

nml_naive, nml_myguess, cl_model, negloglike = np.array(nml_naive), np.array(nml_myguess), np.array(cl_model), np.array(negloglike)

np.argmin(negloglike + cl_model)
np.argmin(negloglike + cl_model + nml_myguess)
np.argmin(negloglike + cl_model + nml_naive)
np.argmin(negloglike + nml_myguess)
np.argmin(negloglike + nml_naive)

fig, ax = plt.subplots(figsize=(5, 2.7), layout = 'constrained')
ax.plot(path["ccp_alphas"], test_scores, label="accuracy")
ax.plot(path["ccp_alphas"], (negloglike + nml_naive + cl_model) / np.max(negloglike + nml_naive + cl_model),
        label="MDL score scaled")
ax.set_xlabel('complexity cost pruning parameter alpha')
ax.legend()