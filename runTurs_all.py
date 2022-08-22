from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from DataInfo import *
from sklearn.model_selection import KFold
from newRuleset import *
from utils_pred import *
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc, average_precision_score, f1_score, confusion_matrix


datasets_without_header_row = ["waveform", "backnote", "chess", "contracept", "iris", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine", "breast"]
datasets_with_header_row = ["avila", "anuran", "diabetes", "sepsis"]


beam_width = 20
num_cut_numeric = 100

for data_name in [datasets_with_header_row, datasets_without_header_row]:
    data_path = "datasets/" + data_name + ".csv"

    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        # sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
        print(data_name, "not in the folder!")

    print("Running TURS on: " + data_path)


    kf = KFold(n_splits=10, shuffle=True, random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
    kfold = kf.split(X=d)

    kfold_list = list(kfold)

    for fold in kfold_list:
        dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
        dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

        le = OrdinalEncoder(dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)
        for icol, tp in enumerate(dtrain.dtypes):
            if tp != float:
                feature_ = dtrain.iloc[:, icol].to_numpy()
                feature_ = feature_.reshape(-1, 1)

                feature_test = dtest.iloc[:, icol].to_numpy()
                feature_test = feature_test.reshape(-1, 1)

                le.fit(feature_)
                dtrain.iloc[:, icol] = le.transform(feature_).reshape(1, -1)[0]
                dtest.iloc[:, icol] = le.transform(feature_test).reshape(1, -1)[0]



        data_info = DataInfo(data=dtrain, max_bin_num=20)

        # Init the Rule, Elserule, Ruleset, ModelingGroupSet, ModelingGroup;
        ruleset = Ruleset(data_info=data_info, features=data_info.features, target=data_info.target)

        # Grow rules;
        ruleset.build(max_iter=1000, beam_width=beam_width, candidate_cuts=data_info.candidate_cuts)

        len(ruleset.rules)
        pruned_ruleset = ruleset.self_prune()


        X_test = dtest.iloc[:, :dtest.shape[1]-1].to_numpy()
        y_test = dtest.iloc[:, dtest.shape[1]-1].to_numpy()
        test_p = get_test_p(pruned_ruleset, X_test)

        if len(test_p[0]) == 2:
            roc_auc = roc_auc_score(y_test, test_p[:,1])
        else:
            roc_auc = roc_auc_score(y_test, test_p, average="weighted", multi_class="ovr")

        print("roc_auc: ", roc_auc)
