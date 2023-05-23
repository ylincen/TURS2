import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc

from turs2.Ruleset import *
from turs2.utils_predict import *
from turs2.ModelEncoding import *
from turs2.DataEncoding import *
from turs2.utils_readable import *

# Global parameter to control
retrain_model = False

# get the columns of the standardized data
X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
colnames = X.columns

# get the no_scale data and choose the same columns as the standardized data
X = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_no_scale.csv')
X = X.loc[:, colnames]
y = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')
X_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_test_no_scale.csv')
X_test = X_test.loc[:, colnames]
y_test = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_test.csv')

constraints = {}
constraints["cut_option"] = {122: RIGHT_CUT}

if retrain_model:
    beamwidth = 1
    data_info = DataInfo(X=X, y=y, num_candidate_cuts=20, max_rule_length=5, feature_names=X.columns, beam_width=1)

    data_encoding = NMLencoding(data_info)
    model_encoding = ModelEncodingDependingOnData(data_info)
    ruleset = Ruleset(data_info=data_info, data_encoding=data_encoding, model_encoding=model_encoding, constraints=constraints)

    t0 = time.time()
    ruleset.fit(max_iter=1000)
    t1 = time.time() - t0
    print("runtime: ", t1)

    # with open("ruleset_icu_with_constraints.pkl", "wb") as f:
    #     pickle.dump(ruleset, f)
else:
    with open("ruleset_icu.pkl", "rb") as f:
        ruleset = pickle.load(f)

res = predict_ruleset(ruleset, X_test, y_test)
roc_auc = roc_auc_score(y_test, res[0][:, 1])
print("roc_auc on test data: ", roc_auc)

# exclude everything with BE, while keeping the 2nd rule's other literals fixed;
new_ruleset = ruleset.modify_rule_i_other_conditions_kept(rule_to_modify_index=2, cols_to_delete_in_rule=np.arange(54, 71))
new_ruleset.fit()
res_new = predict_ruleset(new_ruleset, X_test, y_test)
new_roc_auc = roc_auc_score(y_test, res_new[0][:, 1])
print("roc_auc on test data: ", new_roc_auc)

# (same as above), while keeping the 2nd rule's other variables which are before the BE literal in the growing process


# (same as above), re-search the rule while not keeping anything from the current rule


