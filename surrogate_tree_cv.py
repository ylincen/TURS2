import numpy as np
import pandas as pd

from nml_regret import *
from utils import calc_probs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


def get_tree_cl(X, y):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    kfold = kf.split(X=X, y=y)

    best_ccp_alpha = []
    for train_index, test_index in kfold:
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        path = clf.cost_complexity_pruning_path(X_train, y_train)

        clfs = []
        for ccp_alpha in path["ccp_alphas"]:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            clf.fit(X_train, y_train)
            clfs.append(clf)

        test_scores = [clf.score(X_test, y_test) for clf in clfs]
        best_ccp_alpha.append( path["ccp_alphas"][np.argmax(test_scores)] )

    avg_best_ccp_alpha = np.mean(best_ccp_alpha)
    best_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=avg_best_ccp_alpha)
    best_clf.fit(X, y)

    num_class = len(np.unique(y))
    which_nodes = best_clf.apply(X)

    negloglike, regret_list = [], []
    for i in np.unique(which_nodes):
        instances_covered = (which_nodes == i)
        y_covered = y[instances_covered]

        p = calc_probs(y_covered, num_class)
        p = p[p != 0]
        negloglike.append(-np.sum(np.log2(p) * p) * np.count_nonzero(instances_covered))

        counts = np.count_nonzero(instances_covered)
        regret_list.append(regret(counts, num_class))
    total_negloglike = np.sum(negloglike)
    total_regret = np.sum(regret_list)

    return total_regret + total_negloglike


if __name__ == "__main__":
    data_path = "datasets/avila.csv"
    d = pd.read_csv(data_path)

    le = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
    for icol, tp in enumerate(d.dtypes):
        if tp != float:
            feature_ = d.iloc[:, icol].to_numpy()
            if len(np.unique(feature_)) > 20:
                continue
            feature_ = feature_.reshape(-1, 1)

            le.fit(feature_)
            d.iloc[:, icol] = le.transform(feature_).reshape(1, -1)[0]

    X = d.iloc[:, :d.shape[1] - 1].to_numpy()
    y = d.iloc[:, d.shape[1] - 1].to_numpy()

    tree_cl = get_tree_cl(X, y)
    print(tree_cl)
