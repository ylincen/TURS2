import sys

# for mac local
sys.path.extend(['/Users/yanglincen/projects/TURS'])
sys.path.extend(['/Users/yanglincen/projects/TURS/turs2'])
# for DSlab server:
sys.path.extend(['/home/yangl3/projects/turs'])
sys.path.extend(['/home/yangl3/projects/turs/turs2'])

import numpy as np
import pandas as pd
import copy
import time
import cProfile
from datetime import datetime
# from line_profiler import LineProfiler
import cProfile, pstats

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *

from turs2.exp_utils import *
from turs2.exp_predictive_perf import *

np.seterr(all='raise')

def generate_spurious_values_dependent(X):
    selected_col = np.random.choice(np.arange(X.shape[1]), 1)

    selected_vals = X[:, selected_col[0]]
    unique_vals = np.unique(selected_vals)

    # flip 10% of the binary values
    if len(unique_vals) == 2:
        assert 0 in unique_vals
        assert 1 in unique_vals
        selected_rows = np.random.choice(np.arange(X.shape[0]), size=np.round(0.1 * X.shape[0]).astype(int))
        spurious_values = np.array(selected_vals)
        spurious_values[selected_rows] = 1 - spurious_values[selected_rows]
    else:
        std_ = np.std(selected_vals) * 0.2
        spurious_values = selected_vals + np.random.normal(0, std_, X.shape[0])
    return spurious_values

def generate_spurious_values_indep(X):
    # flip a coin to decide between binary/continuous
    coin = np.random.choice([0, 1], size=1)
    if coin == 0:
        spurious_values = np.random.choice([0, 1], size=X.shape[0], replace=True)
    else:
        spurious_values = np.random.normal(size=X.shape[0])
    return spurious_values


if __name__ == "__main__":
    exp_res_alldata = []
    date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_without_header_row = ["chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                                       "magic", "car", "tic-tac-toe", "wine"]
    datasets_with_header_row = ["avila", "anuran", "diabetes"]

    datasets_with_header_row = datasets_with_header_row + ["Vehicle", "DryBeans"]
    datasets_without_header_row = datasets_without_header_row + ["glass", "pendigits", "HeartCleveland"]

    for data_name in datasets_with_header_row + datasets_without_header_row:

        d = read_data(data_name, datasets_without_header_row=datasets_without_header_row,
                      datasets_with_header_row=datasets_with_header_row)
        d = preprocess_data(d)
        X = d.iloc[:, :d.shape[1] - 1].to_numpy()
        y = d.iloc[:, d.shape[1] - 1].to_numpy()

        num_spurious = np.max([5, np.round(0.5 * d.shape[1])])
        num_spurious = np.min([num_spurious, 50])
        num_spurious = num_spurious.astype(int)
        X_spurious = np.zeros((X.shape[0], num_spurious), dtype=float)
        for icol in range(num_spurious):
            # spurious_values = generate_spurious_values_dependent(X)
            np.random.seed(icol)

            spurious_values = generate_spurious_values_indep(X)

            X_spurious[:, icol] = spurious_values

        combined_array = np.hstack((X, X_spurious, y.reshape(-1, 1)))
        colnames = d.columns.tolist()[:X.shape[1]] + ["spurious_" + str(i) for i in range(num_spurious)] + ["Y"]
        d_spurious = pd.DataFrame(combined_array, columns = colnames)
        d_spurious["Y"] = d_spurious["Y"].astype(int)

        folder_name = "../datasets_spurious_indep/"

        # create the folder if not exsiting
        os.makedirs(folder_name, exist_ok=True)
        d_spurious.to_csv(folder_name + data_name + ".csv", index=False)

