import copy
import pickle
import sys
# sys.path.extend(['/Users/yanglincen/projects/TURS'])
sys.path.extend(['/home/yangl3/projects/TURS'])

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

with open("./avila_ruleset.pickle", "rb") as f:
    ruleset_obj = pickle.load(f)

ruleset = ruleset_obj["ruleset"]
ruleset.fit()
