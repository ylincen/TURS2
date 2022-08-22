import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import LabelEncoder
from DataInfo import *
from sklearn.model_selection import KFold
from newRuleset import *
from newRule import *
from newModelingGroups import *

# import dataset and pre-precess "string/char" categorical variables;
d = pd.read_csv("datasets/iris.csv", header=None)
d = pd.read_csv("datasets/diabetes.csv")
d = pd.read_csv("datasets/magic.csv", header=None)
le = LabelEncoder()
for icol, tp in enumerate(d.dtypes):
    if tp != float:
        le.fit(d.iloc[:, icol])
        d.iloc[:, icol] = le.transform(d.iloc[:, icol])

# train/test split
kf = KFold(n_splits=10, shuffle=True, random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=d)
kfold_list = list(kfold)

dtrain = d.iloc[kfold_list[0][0], :]
dtest = d.iloc[kfold_list[0][1], :]

# Obtain DataInfo object
data_info = DataInfo(data=dtrain)

# Init the Rule, Elserule, Ruleset, ModelingGroupSet, ModelingGroup;
ruleset = Ruleset(data_info=data_info, features=data_info.features, target=data_info.target)

# Grow rules;
ruleset.build(max_iter=1000, beam_width=1, candidate_cuts=data_info.candidate_cuts)
