from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import nml_regret
from DataInfo import *
from sklearn.model_selection import KFold, StratifiedKFold
from newRuleset import *
from utils_pred import *
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc, average_precision_score, f1_score, confusion_matrix


datasets_without_header_row = ["waveform", "backnote", "chess", "contracept", "iris", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine", "breast", "transfusion", "poker"]
datasets_with_header_row = ["avila", "anuran", "diabetes", "sepsis"]

# if len(sys.argv) == 4:
#     data_given = sys.argv[1]
#     beam_width = int(sys.argv[2])
#     num_cut_numeric = int(sys.argv[3])
# else:
#     data_given = "tic-tac-toe"
#     beam_width = 5
#     num_cut_numeric = 100

# data_name = data_given
# data_path = "datasets/" + data_given + ".csv"

# if data_name in datasets_without_header_row:
#     d = pd.read_csv(data_path, header=None)
# elif data_name in datasets_with_header_row:
#     d = pd.read_csv(data_path)
# else:
#     # sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
#     print(data_name, "not in the folder!")

data_path = "xml_challenge/heloc_dataset_v1.csv"
print("Running TURS on: " + data_path)
d = pd.read_csv(data_path)
num_cut_numeric = 100
beam_width = 1



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
X = d.iloc[:, 1:].to_numpy()
y = d.iloc[:, 0].to_numpy()
kfold = kf.split(X=X, y=y)

kfold_list = list(kfold)

for fold in range(1):
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

    X_train = dtrain.iloc[:, 1:dtrain.shape[1]].to_numpy()
    y_train = dtrain.iloc[:, 0].to_numpy()

    data_info = DataInfo(data=dtrain, features=X_train, target=y_train, max_bin_num=num_cut_numeric)

    # Init the Rule, Elserule, Ruleset, ModelingGroupSet, ModelingGroup;
    ruleset = Ruleset(data_info=data_info, features=data_info.features, target=data_info.target)

    # Grow rules;
    ruleset.build(max_iter=1000, beam_width=beam_width, candidate_cuts=data_info.candidate_cuts, print_or_not=True)

    len(ruleset.rules)
    pruned_ruleset = ruleset.self_prune()


    X_test = dtest.iloc[:, 1:].to_numpy()
    y_test = dtest.iloc[:, 0].to_numpy()
    test_p = get_test_p(pruned_ruleset, X_test)

    if len(test_p[0]) == 2:
        roc_auc = roc_auc_score(y_test, test_p[:,1])
    else:
        roc_auc = roc_auc_score(y_test, test_p, average="weighted", multi_class="ovr")

    print("roc_auc: ", roc_auc)
    y_pred = np.argmax(test_p, axis=1)
    acc = np.mean(y_pred == y_test)
    print("ACC: ", acc)

    train_p = get_test_p(pruned_ruleset, X_train)
    if len(train_p[0]) == 2:
        roc_auc_tr = roc_auc_score(y_train, train_p[:,1])
    else:
        roc_auc_tr = roc_auc_score(y_train, train_p, average="weighted", multi_class="ovr")

    print("roc_auc for training set: ", roc_auc_tr)
    y_pred_train = np.argmax(train_p, axis=1)
    acc_tr = np.mean(y_pred_train == y_train)
    print("ACC: ", acc_tr)

    test_bool_all = []
    for rule in ruleset.rules:
        test_bool_all.append(test_bool(rule, X_test))

    for i, bool in enumerate(test_bool_all):
        print(calc_probs(y_test[bool], 2), np.count_nonzero(bool), ruleset.rules[i].prob)

    # # analyse the results
    # negloglike, reg = 0, 0
    # for rule in ruleset.rules:
    #     p = calc_probs(y_train[rule.indices_excl_overlap], data_info.num_class)
    #     p = p[p != 0]
    #     negloglike += len(rule.indices_excl_overlap) * np.sum(-p*np.log2(p))
    #     reg += regret(len(rule.indices_excl_overlap), data_info.num_class)
