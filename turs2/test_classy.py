import numpy as np
import copy

import time
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from rulelist import RuleListClassifier


def generate_data(nrow, ncol):
    X = np.random.randint(2, size=(nrow, ncol)).astype(bool)
    X[:, 0] = np.random.choice([False, True], size=nrow, p=[0.8, 0.2])
    p1 = 0.7  # probability of Y = 0
    p0 = 0.9

    rule = X[:, 0]
    coverage = np.sum(rule)
    y = np.zeros(nrow, dtype=bool)
    y[rule] = np.random.choice([False, True], size=coverage, p=[p1, 1-p1])

    p_else = (p0 * nrow - p1 * coverage) / (nrow - coverage)
    y[~rule] = np.random.choice([False, True], size=nrow - coverage, p=[p_else, 1-p_else])
    X = X.astype(int)
    y = y.astype(int)

    return [X, y]

# np.random.seed(1)
nrow = 10000
ncol = 100
X, y = generate_data(nrow, ncol)
X_test, y_test = generate_data(nrow, ncol)

model = RuleListClassifier(discretization="static")
model.fit(X, y)
y_pred = model.predict_proba(pd.DataFrame(X_test))
auc1 = roc_auc_score(y_test, y_pred[:,1])

print("classy on simulation: ", auc1)
print(model)
