import pandas as pd
import numpy as np

# from turs2.DataInfo import *

X = pd.read_csv(
    r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_StandardScaler_meanimputation_missing_features_dropped.csv')
y = pd.read_csv(
    r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

Xnoscale = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\X_train_no_scale.csv')
ynoscale = pd.read_csv(r'\\vf-DataSafe\DataSafe$\div0\ITenDI\Heropname_1136\Files_Lincen_Siri\Processed datasets\2020\Readmission\y_train.csv')

Xnoscale = Xnoscale.loc[:, X.columns]

print(X.shape)
print(Xnoscale.shape)

# # basic checks
# np.array_equal(y, ynoscale)
#
# x0 = (X.iloc[:, 0] - np.mean(X.iloc[:, 0])) / np.std(X.iloc[:, 0])
# print(max(abs(x0.to_numpy() - X.iloc[:, 0].to_numpy())))
#
# x0_scale = (Xnoscale.iloc[:, 0] - np.mean(Xnoscale.iloc[:, 0])) / np.std(Xnoscale.iloc[:, 0])
# max(abs(x0_scale.to_numpy() - X.iloc[:, 0].to_numpy()))
#
# data_info = DataInfo(X=X, y=y, num_candidate_cuts=20, max_rule_length=5, feature_names=X.columns, beam_width=1)
# data_info_noscale = DataInfo(X=Xnoscale, y=ynoscale, num_candidate_cuts=20,
#                              max_rule_length=5, feature_names=Xnoscale.columns, beam_width=1)
#
# l = []
# for icol in range(X.shape[1]):
#     if X.iloc[:, icol].dtype == 'int64':
#         print(np.unique(X.iloc[:, icol]))
#     else:
#         l.append(len(np.unique(X.iloc[:, icol])))
#
# diffs = []
# for icol in range(X.shape[1]):
#     if X.iloc[:, icol].dtype == 'int64':
#         continue
#     x0 = (X.iloc[:, icol] - np.mean(X.iloc[:, icol])) / np.std(X.iloc[:, icol])
#     diffs.append(max(abs(x0.to_numpy() - X.iloc[:, icol].to_numpy())))
#
# diffs_noscale = []
# for icol in range(Xnoscale.shape[1]):
#     if Xnoscale.iloc[:, icol].dtype == 'int64':
#         continue
#     x0 = (Xnoscale.iloc[:, icol] - np.mean(Xnoscale.iloc[:, icol])) / np.std(Xnoscale.iloc[:, icol])
#     diffs_noscale.append(max(abs(x0.to_numpy() - X.iloc[:, icol].to_numpy())))
#
#
# # for icol in range(data_info.ncol):
# #     candidate_cut_icol = data_info.candidate_cuts[icol]
# #     candidate_cut_icol_noscale = data_info_noscale.candidate_cuts[icol]
# #
# #     if len(candidate_cut_icol_noscale) != len(candidate_cut_icol):
# #         print(icol, candidate_cut_icol, candidate_cut_icol_noscale)
# #
# # # x_icol = X.iloc[:, icol]
# # # x_icol_noscale = Xnoscale.iloc[:, icol]
# #
