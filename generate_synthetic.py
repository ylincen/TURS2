import numpy as np
import sklearn
import sys
from DataInfo import *
from sklearn.model_selection import KFold, StratifiedKFold
from newRuleset import *
from utils_pred import *
import matplotlib.pyplot as plt

# generate n1, ..., n5, such that \sum n_i = n
np.random.seed(1)
n = 1000
x_paras = np.random.dirichlet(alpha=[2, 2, 4, 3, 32], size=None)
ns = np.array(n * x_paras, dtype=int)

real_n = np.sum(ns)

p0 = 0.07  # population/default empirical probability
remaining_p = (p0  * real_n - np.sum(np.array([0.15, 0.2, 0.12, 0.1]) * ns[:4])) / ns[4]
y_paras = np.array([0.15, 0.2, 0.12, 0.1, remaining_p])

# area 1: [0, 0.4] x [0.75, 1]
x1_area1 = np.random.uniform(low=0, high=0.4, size=ns[0])
x2_area1 = np.random.uniform(low=0.75, high=1, size=ns[0])

# area 2: [0, 0.1] x [0.1, 0.2]
x1_area2 = np.random.uniform(low=0, high=0.1, size=ns[1])
x2_area2 = np.random.uniform(low=0.1, high=0.2, size=ns[1])

# area 3: [0.3, 1] x [0, 0.5]
x1_area3 = np.random.uniform(low=0.3, high=1, size=ns[2])
x2_area3 = np.random.uniform(low=0, high=0.5, size=ns[2])

# area 4: [0.85, 1] x [0.7, 0.8]
x1_area4 = np.random.uniform(low=0.85, high=1, size=ns[3])
x2_area4 = np.random.uniform(low=0.7, high=0.8, size=ns[3])

# area 5: the remaining
x1_area5, x2_area5 = [], []
for iter in range(real_n * 100):
    x1_area5_individual = np.random.uniform(low=0, high=1, size=1)
    x2_area5_individual = np.random.uniform(low=0, high=1, size=1)

    if len(x1_area5) >= ns[4] or len(x2_area5) >= ns[4]:
        break

    if x1_area5_individual < 0.4 and x1_area5_individual > 0 and x2_area5_individual < 1 and x2_area5_individual > 0.75:
        continue
    elif x1_area5_individual < 0.1 and x1_area5_individual > 0 and x2_area5_individual < 0.2 and x2_area5_individual > 0.1:
        continue
    elif x1_area5_individual < 1 and x1_area5_individual > 0.3 and x2_area5_individual < 0.5 and x2_area5_individual > 0:
        continue
    elif x1_area5_individual < 1 and x1_area5_individual > 0.85 and x2_area5_individual < 0.8 and x2_area5_individual > 0.7:
        continue
    else:
        x1_area5.append(x1_area5_individual[0])
        x2_area5.append(x2_area5_individual[0])
else:
    sys.exit("Error: try bigger loop number")

x1_area5 = np.array(x1_area5)
x2_area5 = np.array(x2_area5)

y_area1 = np.append(np.repeat(1, round(y_paras[0] * ns[0])), np.repeat(0, ns[0] - round(y_paras[0] * ns[0])))
y_area2 = np.append(np.repeat(1, round(y_paras[1] * ns[1])), np.repeat(0, ns[1] - round(y_paras[1] * ns[1])))
y_area3 = np.append(np.repeat(1, round(y_paras[2] * ns[2])), np.repeat(0, ns[2] - round(y_paras[2] * ns[2])))
y_area4 = np.append(np.repeat(1, round(y_paras[3] * ns[3])), np.repeat(0, ns[3] - round(y_paras[3] * ns[3])))
y_area5 = np.append(np.repeat(1, round(y_paras[4] * ns[4])), np.repeat(0, ns[4] - round(y_paras[4] * ns[4])))


data = np.array((np.hstack([x1_area1, x1_area2, x1_area3, x1_area4, x1_area5]),
                 np.hstack([x2_area1, x2_area2, x2_area3, x2_area4, x2_area5]),
                 np.hstack([y_area1, y_area2, y_area3, y_area4, y_area5])))
data = data.T

X = np.array([np.hstack([x1_area1, x1_area2, x1_area3, x1_area4, x1_area5]),
              np.hstack([x2_area1, x2_area2, x2_area3, x2_area4, x2_area5])]).T
y = np.hstack([y_area1, y_area2, y_area3, y_area4, y_area5])


data_info = DataInfo(data=data, features=X, target=y, max_bin_num=40)

# Init the Rule, Elserule, Ruleset, ModelingGroupSet, ModelingGroup;
ruleset = Ruleset(data_info=data_info, features=data_info.features, target=data_info.target,
                  number_of_init_rules=1)

# Grow rules;
ruleset.build(max_iter=1000, beam_width=1, candidate_cuts=data_info.candidate_cuts, print_or_not=True)











