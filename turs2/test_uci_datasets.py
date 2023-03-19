import numpy as np
import pandas as pd
import sklearn.datasets
import copy

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *


datasets_without_header_row = ["waveform", "backnote", "chess", "contracept", "iris", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine"]
datasets_with_header_row = ["avila", "anuran", "diabetes"]

for data_name in datasets_without_header_row + datasets_with_header_row:
    if data_name == "avila":
        continue

    data_path = "../datasets/" + data_name + ".csv"
    print("running: ", data_name)
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        sys.exit("error: data name not in the datasets lists that show whether the header should be included!")

    kf = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
    X = d.iloc[:, :d.shape[1]-1].to_numpy()
    y = d.iloc[:, d.shape[1]-1].to_numpy()
    kfold = kf.split(X=X, y=y)

    kfold_list = list(kfold)

    skip_this_data = False
    for fold in range(1):
        dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
        dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])
        le = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
        for icol, tp in enumerate(dtrain.dtypes):
            if icol == dtrain.shape[1] - 1:
                feature_ = dtrain.iloc[:, icol].to_numpy()  # NOTE: feature_ is actually target here!!
                feature_ = feature_.reshape(-1, 1)

                feature_test = dtest.iloc[:, icol].to_numpy()
                feature_test = feature_test.reshape(-1, 1)

                le.fit(feature_)
                dtrain.iloc[:, icol] = le.transform(feature_).reshape(1, -1)[0]
                dtest.iloc[:, icol] = le.transform(feature_test).reshape(1, -1)[0]
                continue

            if tp != float:
                feature_ = dtrain.iloc[:, icol].to_numpy()
                if len(np.unique(feature_)) > 5:
                    feature_ = feature_.reshape(-1, 1)

                    feature_test = dtest.iloc[:, icol].to_numpy()
                    feature_test = feature_test.reshape(-1, 1)

                    le.fit(feature_)
                    dtrain.iloc[:, icol] = le.transform(feature_).reshape(1, -1)[0]
                    dtest.iloc[:, icol] = le.transform(feature_test).reshape(1, -1)[0]
                else:
                    skip_this_data = True
                    break
        if skip_this_data:
            print("Skip " + data_name + " due to non-numeric features!")
            break

        X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
        y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
        X_test = dtest.iloc[:, :-1].to_numpy()
        y_test = dtest.iloc[:, -1].to_numpy()

        data_info = DataInfo(X=X_train, y=y_train, num_candidate_cuts=10, max_rule_length=10, feature_names=dtrain.columns[:-1], beam_width=1)
        ruleset = Ruleset(data_info=data_info)

        # ruleset.fit(max_iter=1000, printing=False)
        # res = predict_ruleset(ruleset, X_test, y_test)
        ruleset.fit_rulelist(max_iter=1000)
        res = predict_rulelist(ruleset, X_test, y_test)

        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, res[0][:, 1])
        else:
            roc_auc = roc_auc_score(y_test, res[0], multi_class="ovr")

        print(roc_auc)
