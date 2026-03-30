import os
import time
import cProfile
from datetime import datetime
from line_profiler import LineProfiler
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

from turs2.data_with_spurious_features import *

np.seterr(all='raise')

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")

data_names = os.listdir("../ADbench_datasets_Classical/")

for data_name in data_names:
    if "npz" not in data_name:
        continue
    print("Running on: ", data_name)
    d_np = np.load("../ADbench_datasets_Classical/" + data_name)
    X = d_np["X"]
    y = d_np["y"]
    d_original = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
    d = preprocess_data(d_original)

    num_spurious = np.max([5, np.round(0.5 * d.shape[1])])
    num_spurious = np.min([num_spurious, 50])
    num_spurious = num_spurious.astype(int)
    X_spurious = np.zeros((X.shape[0], num_spurious), dtype=float)
    for icol in range(num_spurious):
        np.random.seed(icol)

        spurious_values = generate_spurious_values_dependent(X)
        # spurious_values = generate_spurious_values_indep(X)

        X_spurious[:, icol] = spurious_values

    combined_array = np.hstack((X, X_spurious, y.reshape(-1, 1)))
    colnames = d.columns.tolist()[:X.shape[1]] + ["spurious_" + str(i) for i in range(num_spurious)] + ["Y"]
    d_spurious = pd.DataFrame(combined_array, columns = colnames)
    d_spurious["Y"] = d_spurious["Y"].astype(int)

    folder_name = "../adbench_datasets_spurious_dep/"

    # create the folder if not exsiting
    os.makedirs(folder_name, exist_ok=True)
    d_spurious.to_csv(folder_name + data_name + ".csv", index=False)