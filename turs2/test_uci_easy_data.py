import copy

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.model_selection import StratifiedKFold, train_test_split

from turs2.DataInfo import *
from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.ensemble import RandomForestClassifier

datasets_without_header_row = ["waveform", "backnote", "chess", "contracept", "iris", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine"]
datasets_with_header_row = ["avila", "anuran", "diabetes"]

binary_class_datasets = ["backnote", "chess", "diabetes", "ionosphere", "magic", "tic-tac-toe"]

for data_name in datasets_without_header_row + datasets_with_header_row:
    if data_name != "diabetes":
        continue
    if data_name not in binary_class_datasets:
        continue
    if data_name == "magic":
        continue

    data_path = "../datasets/" + data_name + ".csv"
    print("running: ", data_name)
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        sys.exit("error: data name not in the datasets lists that show whether the header should be included!")

    if data_name == "waveform":
        d = d.iloc[:, 1:]
        d.columns = np.arange(d.shape[1])

    le_target = LabelEncoder()
    le_target.fit(d.iloc[:, d.shape[1] - 1])
    d.iloc[:, d.shape[1] - 1] = le_target.transform(d.iloc[:, d.shape[1] - 1])

    cols_to_encode_with_onehot = []
    for i_col in range(d.shape[1] - 1):
        if (d.dtypes[i_col] == "int64" or d.dtypes[i_col] == "int32") and len(np.unique(d.iloc[:, i_col])) < 5 \
                and len(np.unique(d.iloc[:, i_col])) > 2:
            cols_to_encode_with_onehot.append(i_col)

    d = pd.get_dummies(d, columns=[d.columns[kkkk] for kkkk in cols_to_encode_with_onehot])  # handle integers with a few unique values
    d = pd.get_dummies(d)  # handle object, string, or category dtype



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

        X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
        y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
        X_test = dtest.iloc[:, :-1].to_numpy()
        y_test = dtest.iloc[:, -1].to_numpy()

        rf = RandomForestClassifier(n_estimators=200, n_jobs=1, oob_score=True, min_samples_leaf=10)
        rf.fit(X_train, y_train)

        data_info = DataInfo(X=X_train, y=y_train, num_candidate_cuts=100, max_rule_length=5,
                             feature_names=dtrain.columns[:-1], beam_width=1, X_test=X_test, y_test=y_test,
                             dataset_name=data_name, log_learning_process=True, rf_oob_decision_=rf.oob_decision_function_[:,1])

        data_encoding = NMLencoding(data_info)
        model_encoding = ModelEncodingDependingOnData(data_info)
        ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)

        ruleset.fit(max_iter=1000, printing=True)
        res = predict_ruleset(ruleset, X_test, y_test)
        # ruleset.fit_rulelist(max_iter=1000)
        # res = predict_rulelist(ruleset, X_test, y_test)

        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y_test, res[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, res, multi_class="ovr")

        print("%%%%%%%%%%%%ROC AUC on test data, rule set: ", roc_auc, "\n\n")



y_pred_p = rf.predict_proba(X_test)
y_pred = rf.predict(X_test)

print("=====Random Forest results:")
if len(np.unique(y_train)) > 2:
    y_pred_p_train = rf.predict_proba(X_train)
    print("training set roc auc: ", roc_auc_score(y_train, y_pred_p_train, multi_class="ovr"))
    print("roc auc on test data: ", roc_auc_score(y_test, y_pred_p, multi_class="ovr"))
    print("oob roc auc: ", roc_auc_score(y_train, rf.oob_decision_function_, multi_class="ovr"))

    else_rule_cover = ruleset.uncovered_indices
    oob_df_else_rule = np.array(rf.oob_decision_function_)
    oob_df_else_rule[else_rule_cover] = np.median(oob_df_else_rule[else_rule_cover])
    print("oob roc auc after putting else rule in: ", roc_auc_score(y_train, oob_df_else_rule, multi_class="ovr"))
else:
    y_pred_p_train = rf.predict_proba(X_train)
    print("training set roc auc: ", roc_auc_score(y_train, y_pred_p_train[:,1]))
    print("roc auc on test data: ", roc_auc_score(y_test, y_pred_p[:,1]))
    print("oob roc auc: ", roc_auc_score(y_train, rf.oob_decision_function_[:,1]))

    else_rule_cover = ruleset.uncovered_indices
    oob_df_else_rule = np.array(rf.oob_decision_function_)
    oob_df_else_rule[else_rule_cover] = np.median(oob_df_else_rule[else_rule_cover])
    print("oob roc auc after putting else rule in: ", roc_auc_score(y_train, oob_df_else_rule[:,1]))
