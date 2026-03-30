import sys
import platform

# TODO: make this part nicer by using relative path
# This is what I did on the TURS2 github repo (so the "submitted code") [but why is it different?? FUCK no idea]
if platform.system() == "Darwin":
    # for mac local
    sys.path.extend(['/Users/yanglincen/projects/TURS'])
    sys.path.extend(['/Users/yanglincen/projects/TURS/turs2'])

    data_folder_name = "../datasets_for_abalation/processed/"
elif platform.system() == "Linux":
    # for DSlab server:
    sys.path.extend(['/home/yangl3/projects/turs'])
    sys.path.extend(['/home/yangl3/projects/turs/turs2'])

    data_folder_name = "/data/yangl3/datasets_jmlr_ablation/processed/"

elif platform.system() == "Windows":
    sys.exit("Windows system is not supported yet.")


from turs2.exp_predictive_perf import *

np.seterr(all='raise')

exp_res_alldata = []
date_and_time = datetime.now().strftime("%Y%m%d_%H%M%S")
datasets_without_header_row = []
datasets_with_header_row = ["fico", "naticusdroid", "smoking", "TUANDROMD", "airline"]

if len(sys.argv) > 1: # if the data_name is given
    assert len(sys.argv) == 4, "need to provide 3 arguments: data_name, validity_check_, not_use_excl_"
    test_mode = False
    data_name = sys.argv[1]
    fold_given = None
    validity_check_ = sys.argv[2]
    not_use_excl_ = sys.argv[3]
else:
    test_mode = True
    print("test mode")
    data_name = "fico"
    fold_given = 0
    validity_check_ = "either"
    not_use_excl_ = True

d = read_data(data_name, datasets_without_header_row=datasets_without_header_row,
              datasets_with_header_row=datasets_with_header_row, folder_name=data_folder_name)
d = preprocess_data(d, threshold_categorical=2)

X = d.iloc[:, :d.shape[1] - 1].to_numpy()
y = d.iloc[:, d.shape[1] - 1].to_numpy()

kf = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=2)  # can also use sklearn.model_selection.StratifiedKFold
kfold = kf.split(X=X, y=y)
kfold_list = list(kfold)

# options_not_use_excl_ = [True, False]
# options_validity_check_ = ["none", "either"]

# counter_ = 0
# for not_use_excl_, validity_check_ in zip(options_not_use_excl_, options_validity_check_):
#     if len(sys.argv) < 2:
#         print("test mode, only test: not_use_excl_= False & validity_check_= either")
#         not_use_excl_ = False
#         validity_check_ = "either"
#         if counter_ > 0:
#             break

for fold in range(5):
    if fold_given is not None and fold != fold_given:
        continue
    print("running: ", data_name, "; fold: ", fold)
    dtrain = copy.deepcopy(d.iloc[kfold_list[fold][0], :])
    dtest = copy.deepcopy(d.iloc[kfold_list[fold][1], :])

    X_train = dtrain.iloc[:, :dtrain.shape[1]-1].to_numpy()
    y_train = dtrain.iloc[:, dtrain.shape[1]-1].to_numpy()
    X_test = dtest.iloc[:, :-1].to_numpy()
    y_test = dtest.iloc[:, -1].to_numpy()

    start_time = time.time()
    alg_config = AlgConfig(
        num_candidate_cuts=20, max_num_rules=500, max_grow_iter=500, num_class_as_given=None,
        beam_width=10,
        log_learning_process=False,
        dataset_name=None, X_test=None, y_test=None,
        rf_assist=False, rf_oob_decision_function=None,
        feature_names=["X" + str(i) for i in range(X.shape[1])],
        beamsearch_positive_gain_only=False, beamsearch_normalized_gain_must_increase_comparing_rulebase=False,
        beamsearch_stopping_when_best_normalized_gain_decrease=False,
        validity_check=validity_check_, rerun_on_invalid=False, rerun_positive_control=False,
        min_sample_each_rule=1
    )
    data_info = DataInfo(X=X_train, y=y_train, beam_width=None, alg_config=alg_config,
                         not_use_excl_=not_use_excl_)

    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding)
    ruleset.fit(max_iter=1000, printing=True)
    end_time = time.time()

    ## ROC_AUC and log-loss
    exp_res = calculate_exp_res(ruleset, X_test, y_test, X_train, y_train, data_name, fold, start_time, end_time)
    exp_res["not_use_excl_"] = not_use_excl_
    exp_res["validity_check_"] = validity_check_

    exp_res_alldata.append(exp_res)
exp_res_df = pd.DataFrame(exp_res_alldata)

if test_mode:
    folder_name = "TEST_MODE_exp_ablation_new_datasetes" + date_and_time[:8]
else:
    folder_name = "NEWEXP_ablation_new_datasetes" + date_and_time[:8]
    os.makedirs(folder_name, exist_ok=True)
    res_file_name = "./" + folder_name + "/" + date_and_time + "_" + data_name + "_" + str(not_use_excl_) + "_" + str(validity_check_)+ ".csv"
    exp_res_df.to_csv(res_file_name, index=False)

aa1 = np.array([a[0] for a in exp_res["rules_prob_test"]]) == 0
aa2 = np.array([a[1] for a in exp_res["rules_prob_test"]]) == 0
print(np.where(aa1 & aa2)[0])

rule_prob_diff = np.round(np.array([a[0] for a in exp_res["rules_prob_test"]]) - np.array([a[0] for a in exp_res["rules_prob_train"]]), 2)
coverage_test = exp_res["cover_matrix_test"].sum(axis=0)

rule_prob_test = np.round(np.array([a[0] for a in exp_res["rules_prob_test"]]), 2)
rule_prob_train = np.round(np.array([a[0] for a in exp_res["rules_prob_train"]]), 2)

[print(a,b, c) for a, b, c in zip(rule_prob_train, rule_prob_test, coverage_test)]

# get the calibration curve
